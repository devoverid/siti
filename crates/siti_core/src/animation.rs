//! Meme captions for animations: decodes a silent H.264 MP4 or a GIF, draws the captions on
//! every frame and re-encodes H.264 MP4. Pure Rust plus OpenH264 built from source.

use std::io::Cursor;

use bytes::Bytes;
use image::{codecs::gif::GifDecoder, AnimationDecoder, ImageDecoder};
use mp4::{
    AvcConfig, MediaConfig, MediaType, Mp4Config, Mp4Reader, Mp4Sample, Mp4Writer, TrackConfig,
};
use openh264::{
    decoder::{Decoder, DecoderConfig, Flush},
    encoder::{BitRate, Encoder, EncoderConfig, FrameRate, FrameType, IntraFramePeriod, RateControlMode},
    formats::{RgbSliceU8, YUVBuffer, YUVSource},
    OpenH264API,
};

use crate::{
    error::SitiError,
    meme::{self, Captions},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationFormat {
    Mp4,
    Gif,
}

/// Bot API getFile download limit.
pub const MAX_INPUT_BYTES: u32 = 20 * 1024 * 1024;
/// Telegram sendAnimation upload limit.
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;
pub const MAX_FRAMES: u32 = 600;
pub const MAX_SIDE: u32 = 1280;
const MIN_SIDE: u32 = 16;
/// GIF delays below this are treated as "unspecified", like browsers do.
const MIN_GIF_DELAY_MS: u32 = 20;
const DEFAULT_GIF_DELAY_MS: u32 = 100;

/// Captions every frame of a silent H.264 MP4 or a GIF and returns H.264 MP4. CPU-bound.
pub fn add_text(data: &[u8], format: AnimationFormat, top: &str, bottom: &str) -> Result<Vec<u8>, SitiError> {
    // Fail on bad text before touching the media.
    meme::validate(top, bottom)?;
    match format {
        AnimationFormat::Mp4 => caption_mp4(data, top, bottom),
        AnimationFormat::Gif => caption_gif(data, top, bottom),
    }
}

/// [`add_text`] on the blocking thread pool.
pub async fn add_text_async(data: Vec<u8>, format: AnimationFormat, top: String, bottom: String) -> Result<Vec<u8>, SitiError> {
    tokio::task::spawn_blocking(move || add_text(&data, format, &top, &bottom))
        .await
        .map_err(|e| SitiError::Generate(format!("animation rendering failed: {e}")))?
}

fn check_size(w: u32, h: u32) -> Result<(), SitiError> {
    if (MIN_SIDE..=MAX_SIDE).contains(&w) && (MIN_SIDE..=MAX_SIDE).contains(&h) {
        Ok(())
    } else {
        Err(SitiError::Unsupported(format!("resolution {w}x{h} outside {MIN_SIDE}..={MAX_SIDE}")))
    }
}

fn check_frames(n: u32) -> Result<(), SitiError> {
    if n > MAX_FRAMES {
        Err(SitiError::Unsupported(format!("more than {MAX_FRAMES} frames")))
    } else {
        Ok(())
    }
}

/// Strips a leading Annex B start code (`00 00 01` or `00 00 00 01`).
fn strip_start_code(nal: &[u8]) -> &[u8] {
    if let Some(rest) = nal.strip_prefix(&[0, 0, 0, 1]) {
        rest
    } else if let Some(rest) = nal.strip_prefix(&[0, 0, 1]) {
        rest
    } else {
        nal
    }
}

/// Encodes RGB frames to H.264 and muxes them into an MP4, created lazily on the first frame.
struct Mp4Out {
    w: u32,
    h: u32,
    yuv: YUVBuffer,
    encoder: Encoder,
    writer: Option<Mp4Writer<Cursor<Vec<u8>>>>,
    clock_ms: u64,
}

impl Mp4Out {
    fn new(w: u32, h: u32, fps: f32) -> Result<Self, SitiError> {
        let bps = (w * h).saturating_mul(4).clamp(500_000, 4_000_000);
        let config = EncoderConfig::new()
            .skip_frames(false)
            .rate_control_mode(RateControlMode::Quality)
            .bitrate(BitRate::from_bps(bps))
            .max_frame_rate(FrameRate::from_hz(fps.clamp(1.0, 60.0)))
            .intra_frame_period(IntraFramePeriod::from_num_frames(60));
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config).map_err(internal)?;
        Ok(Self { w, h, yuv: YUVBuffer::new(w as usize, h as usize), encoder, writer: None, clock_ms: 0 })
    }

    /// `rgb` is tightly packed RGB8 of exactly `w`×`h`.
    fn push(&mut self, rgb: &[u8], ms: u32) -> Result<(), SitiError> {
        self.yuv.read_rgb8(RgbSliceU8::new(rgb, (self.w as usize, self.h as usize)));
        let bs = self.encoder.encode(&self.yuv).map_err(internal)?;
        let is_sync = matches!(bs.frame_type(), FrameType::IDR | FrameType::I);
        let (mut sps, mut pps) = (None, None);
        let mut sample = Vec::new();
        for l in 0..bs.num_layers() {
            let Some(layer) = bs.layer(l) else { continue };
            for n in 0..layer.nal_count() {
                let Some(nal) = layer.nal_unit(n) else { continue };
                let nal = strip_start_code(nal);
                match nal.first().map(|b| b & 0x1F) {
                    Some(7) => sps = sps.or_else(|| Some(nal.to_vec())),
                    Some(8) => pps = pps.or_else(|| Some(nal.to_vec())),
                    Some(_) => {
                        sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                        sample.extend_from_slice(nal);
                    }
                    None => {}
                }
            }
        }
        if self.writer.is_none() {
            let (Some(sps), Some(pps)) = (sps, pps) else {
                return Err(SitiError::Generate("encoder produced no parameter sets".into()));
            };
            let brands = ["isom", "iso2", "avc1", "mp41"].map(|b| b.parse().expect("valid fourcc"));
            let mut writer = Mp4Writer::write_start(
                Cursor::new(Vec::new()),
                &Mp4Config { major_brand: "isom".parse().expect("valid fourcc"), minor_version: 512, compatible_brands: brands.to_vec(), timescale: 1000 },
            )
            .map_err(internal)?;
            let track = TrackConfig::from(MediaConfig::AvcConfig(AvcConfig {
                width: self.w as u16,
                height: self.h as u16,
                seq_param_set: sps,
                pic_param_set: pps,
            }));
            writer.add_track(&track).map_err(internal)?;
            self.writer = Some(writer);
        }
        if sample.is_empty() {
            return Err(SitiError::Generate("encoder produced an empty frame".into()));
        }
        let writer = self.writer.as_mut().expect("created above");
        writer
            .write_sample(1, &Mp4Sample { start_time: self.clock_ms, duration: ms, rendering_offset: 0, is_sync, bytes: Bytes::from(sample) })
            .map_err(internal)?;
        self.clock_ms += ms as u64;
        Ok(())
    }

    fn finish(self) -> Result<Vec<u8>, SitiError> {
        let mut writer = self.writer.ok_or_else(|| SitiError::Generate("no frames decoded".into()))?;
        writer.write_end().map_err(internal)?;
        let out = writer.into_writer().into_inner();
        if out.len() > MAX_OUTPUT_BYTES {
            return Err(SitiError::Unsupported("result too large".into()));
        }
        Ok(out)
    }
}

/// Per-animation state shared by both inputs: crops to even size, captions, encodes.
struct Pipeline<'a> {
    top: &'a str,
    bottom: &'a str,
    fps: f32,
    state: Option<(Captions, Mp4Out)>,
}

impl<'a> Pipeline<'a> {
    fn new(top: &'a str, bottom: &'a str, fps: f32) -> Self {
        Self { top, bottom, fps, state: None }
    }

    /// `rgb` is a tightly packed `w`×`h` frame; it is cropped to even dimensions in place.
    fn frame(&mut self, rgb: &mut [u8], w: u32, h: u32, ms: u32) -> Result<(), SitiError> {
        let (ew, eh) = (w & !1, h & !1);
        if self.state.is_none() {
            check_size(w, h)?;
            self.state = Some((Captions::new(self.top, self.bottom, ew, eh)?, Mp4Out::new(ew, eh, self.fps)?));
        }
        let (captions, out) = self.state.as_mut().expect("initialized above");
        if (out.w, out.h) != (ew, eh) {
            return Err(SitiError::Generate("frame size changed mid-stream".into()));
        }
        let (w, ew, eh) = (w as usize, ew as usize, eh as usize);
        if w != ew {
            for y in 0..eh {
                rgb.copy_within(y * w * 3..y * w * 3 + ew * 3, y * ew * 3);
            }
        }
        let frame = &mut rgb[..ew * eh * 3];
        captions.apply(frame);
        out.push(frame, ms)
    }

    fn finish(self) -> Result<Vec<u8>, SitiError> {
        self.state.ok_or_else(|| SitiError::Generate("no frames decoded".into()))?.1.finish()
    }
}

fn caption_gif(data: &[u8], top: &str, bottom: &str) -> Result<Vec<u8>, SitiError> {
    let decoder = GifDecoder::new(Cursor::new(data)).map_err(internal)?;
    let (w, h) = decoder.dimensions();
    check_size(w, h)?;
    let mut pipeline = Pipeline::new(top, bottom, 30.0);
    let mut rgb = vec![0u8; w as usize * h as usize * 3];
    let mut count = 0;
    for frame in decoder.into_frames() {
        count += 1;
        check_frames(count)?;
        let frame = frame.map_err(internal)?;
        let (n, d) = frame.delay().numer_denom_ms();
        let ms = n / d.max(1);
        let ms = if ms < MIN_GIF_DELAY_MS { DEFAULT_GIF_DELAY_MS } else { ms };
        let buf = frame.into_buffer();
        if buf.dimensions() != (w, h) {
            return Err(SitiError::Generate("GIF frame size differs from the canvas".into()));
        }
        for (dst, src) in rgb.chunks_exact_mut(3).zip(buf.pixels()) {
            let a = src.0[3] as u32;
            for c in 0..3 {
                dst[c] = ((src.0[c] as u32 * a + 255 * (255 - a)) / 255) as u8;
            }
        }
        pipeline.frame(&mut rgb, w, h, ms)?;
    }
    pipeline.finish()
}

fn caption_mp4(data: &[u8], top: &str, bottom: &str) -> Result<Vec<u8>, SitiError> {
    let fps = mp4_fps(data)?;
    let mut pipeline = Pipeline::new(top, bottom, fps);
    decode_mp4(data, |rgb, w, h, ms| pipeline.frame(rgb, w, h, ms))?;
    pipeline.finish()
}

fn mp4_fps(data: &[u8]) -> Result<f32, SitiError> {
    let reader = Mp4Reader::read_header(Cursor::new(data), data.len() as u64).map_err(internal)?;
    let track = h264_track(&reader)?;
    let secs = track.duration().as_secs_f32();
    Ok(if secs > 0.0 { track.sample_count() as f32 / secs } else { 30.0 })
}

fn h264_track<'a, R: std::io::Read + std::io::Seek>(reader: &'a Mp4Reader<R>) -> Result<&'a mp4::Mp4Track, SitiError> {
    reader
        .tracks()
        .values()
        .find(|t| t.media_type().ok() == Some(MediaType::H264))
        .ok_or_else(|| SitiError::Unsupported("no H.264 video track".into()))
}

fn internal(e: impl std::fmt::Display) -> SitiError {
    SitiError::Generate(e.to_string())
}

/// Converts one length-prefixed AVC sample to Annex B, prepending SPS/PPS on sync samples.
fn to_annex_b(sample: &[u8], length_size: usize, is_sync: bool, params: &[Vec<u8>], out: &mut Vec<u8>) -> Result<(), SitiError> {
    const START: [u8; 4] = [0, 0, 0, 1];
    out.clear();
    if is_sync {
        for p in params {
            out.extend_from_slice(&START);
            out.extend_from_slice(p);
        }
    }
    let mut rest = sample;
    while !rest.is_empty() {
        let (len, tail) = rest
            .split_at_checked(length_size)
            .ok_or_else(|| SitiError::Generate("corrupt H.264 sample".into()))?;
        let len = len.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize);
        let (nal, tail) = tail.split_at_checked(len).ok_or_else(|| SitiError::Generate("corrupt H.264 sample".into()))?;
        out.extend_from_slice(&START);
        out.extend_from_slice(nal);
        rest = tail;
    }
    Ok(())
}

/// Decodes every sample of the first H.264 track, calling `sink(rgb, w, h, duration_ms)`.
fn decode_mp4(data: &[u8], mut sink: impl FnMut(&mut [u8], u32, u32, u32) -> Result<(), SitiError>) -> Result<(), SitiError> {
    let mut reader = Mp4Reader::read_header(Cursor::new(data), data.len() as u64).map_err(internal)?;
    let track = h264_track(&reader)?;
    check_frames(track.sample_count())?;
    check_size(track.width() as u32, track.height() as u32)?;
    let id = track.track_id();
    let count = track.sample_count();
    let timescale = track.timescale().max(1) as u64;
    let avcc = &track.trak.mdia.minf.stbl.stsd.avc1.as_ref().ok_or_else(|| internal("missing avc1 box"))?.avcc;
    let length_size = avcc.length_size_minus_one as usize + 1;
    let params: Vec<Vec<u8>> = avcc
        .sequence_parameter_sets
        .iter()
        .chain(&avcc.picture_parameter_sets)
        .map(|n| n.bytes.clone())
        .collect();

    let mut decoder = Decoder::with_api_config(OpenH264API::from_source(), DecoderConfig::new().flush_after_decode(Flush::NoFlush)).map_err(internal)?;
    let mut durations: Vec<u32> = Vec::with_capacity(count as usize);
    let mut emitted = 0usize;
    let mut annex_b = Vec::new();
    let mut rgb = Vec::new();
    let mut emit = |yuv: &openh264::decoder::DecodedYUV<'_>, durations: &[u32]| -> Result<(), SitiError> {
        let (w, h) = yuv.dimensions();
        rgb.resize(w * h * 3, 0);
        yuv.write_rgb8(&mut rgb);
        let ms = durations.get(emitted).or(durations.last()).copied().unwrap_or(100);
        emitted += 1;
        sink(&mut rgb, w as u32, h as u32, ms)
    };
    for sample_id in 1..=count {
        let Some(sample) = reader.read_sample(id, sample_id).map_err(internal)? else { continue };
        durations.push(((sample.duration as u64 * 1000 / timescale) as u32).max(1));
        to_annex_b(&sample.bytes, length_size, sample.is_sync, &params, &mut annex_b)?;
        if let Some(yuv) = decoder.decode(&annex_b).map_err(|e| internal(format!("sample {sample_id}: {e}")))? {
            emit(&yuv, &durations)?;
        }
    }
    for yuv in decoder.flush_remaining().map_err(internal)? {
        emit(&yuv, &durations)?;
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use image::{codecs::gif::GifEncoder, Delay, Frame, RgbaImage};

    fn solid_mp4(w: u32, h: u32, frames: u32, ms: u32) -> Vec<u8> {
        let rgb: Vec<u8> = [40, 120, 200].repeat((w * h) as usize);
        let mut out = Mp4Out::new(w, h, 10.0).unwrap();
        for _ in 0..frames {
            out.push(&rgb, ms).unwrap();
        }
        out.finish().unwrap()
    }

    fn gif(w: u32, h: u32, frames: u32, delay_ms: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            for _ in 0..frames {
                let img = RgbaImage::from_pixel(w, h, image::Rgba([40, 120, 200, 255]));
                enc.encode_frame(Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(delay_ms, 1))).unwrap();
            }
        }
        buf
    }

    fn track_info(mp4: &[u8]) -> (u32, u32, u32, Vec<u32>) {
        let mut r = Mp4Reader::read_header(Cursor::new(mp4), mp4.len() as u64).unwrap();
        assert_eq!(r.tracks().len(), 1);
        let t = h264_track(&r).unwrap();
        let (id, n, w, h) = (t.track_id(), t.sample_count(), t.width() as u32, t.height() as u32);
        let durs = (1..=n).map(|i| r.read_sample(id, i).unwrap().unwrap().duration).collect();
        (n, w, h, durs)
    }

    #[test]
    fn mp4_round_trip_keeps_timing_and_draws_captions() {
        let out = add_text(&solid_mp4(320, 240, 10, 100), AnimationFormat::Mp4, "top text", "bottom text").unwrap();
        let (n, w, h, durs) = track_info(&out);
        assert_eq!((n, w, h), (10, 320, 240));
        assert!((durs.iter().sum::<u32>() as i64 - 1000).abs() <= 10);

        let mut first = None;
        decode_mp4(&out, |rgb, _, _, _| {
            first.get_or_insert_with(|| rgb.to_vec());
            Ok(())
        })
        .unwrap();
        let img = first.unwrap();
        let white = |y0: usize, y1: usize| (y0..y1).any(|y| (0..320).any(|x| img[(y * 320 + x) * 3..][..3].iter().all(|&c| c > 200)));
        assert!(white(0, 72), "top caption missing");
        assert!(white(168, 240), "bottom caption missing");
        assert!(!white(96, 144), "middle must stay untouched");
    }

    #[test]
    fn gif_odd_size_is_cropped_and_delays_kept() {
        let out = add_text(&gif(101, 75, 3, 70), AnimationFormat::Gif, "a", "b").unwrap();
        let (n, w, h, durs) = track_info(&out);
        assert_eq!((n, w, h), (3, 100, 74));
        assert_eq!(durs, vec![70, 70, 70]);
    }

    #[test]
    fn rejects_oversized_gif() {
        let err = add_text(&gif(1282, 16, 1, 100), AnimationFormat::Gif, "a", "").unwrap_err();
        assert!(matches!(err, SitiError::Unsupported(_)), "{err}");
    }

    #[test]
    fn rejects_garbage() {
        assert!(add_text(b"not a video", AnimationFormat::Mp4, "a", "").is_err());
    }
}
