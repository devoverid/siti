//! Meme captions for animations: decodes a silent H.264 MP4 or a GIF, draws the captions on
//! every frame and re-encodes H.264 MP4. Pure Rust plus OpenH264 built from source.

use std::{
    io::Cursor,
    time::{Duration, Instant},
};

use bytes::Bytes;
use image::{codecs::gif::GifDecoder, AnimationDecoder, ImageDecoder};
use mp4::{
    AvcConfig, MediaConfig, MediaType, Mp4Config, Mp4Reader, Mp4Sample, Mp4Writer, TrackConfig,
};
use openh264::{
    decoder::{Decoder, DecoderConfig, Flush},
    encoder::{
        BitRate, Encoder, EncoderConfig, FrameRate, FrameType, IntraFramePeriod, RateControlMode, SpsPpsStrategy,
    },
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
/// Wall-clock budget for one animation; checked once per frame.
const MAX_RENDER_TIME: Duration = Duration::from_secs(60);

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
            .intra_frame_period(IntraFramePeriod::from_num_frames(60))
            // avcC stores only the first SPS/PPS; later IDRs must repeat identical ones.
            .sps_pps_strategy(SpsPpsStrategy::ConstantId);
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config).map_err(internal)?;
        Ok(Self { w, h, yuv: YUVBuffer::new(w as usize, h as usize), encoder, writer: None, clock_ms: 0 })
    }

    /// `rgb` is tightly packed RGB8 of exactly `w`×`h`.
    fn push(&mut self, rgb: &[u8], ms: u32) -> Result<(), SitiError> {
        self.yuv.read_rgb8(RgbSliceU8::new(rgb, (self.w as usize, self.h as usize)));
        let bs = self.encoder.encode(&self.yuv).map_err(internal)?;
        let is_sync = matches!(bs.frame_type(), FrameType::IDR);
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
    /// Encoder rate hint; `None` derives it from the first frame's duration.
    fps: Option<f32>,
    deadline: Instant,
    state: Option<(Captions, Mp4Out)>,
}

impl<'a> Pipeline<'a> {
    fn new(top: &'a str, bottom: &'a str, fps: Option<f32>) -> Self {
        Self { top, bottom, fps, deadline: Instant::now() + MAX_RENDER_TIME, state: None }
    }

    /// `rgb` is a tightly packed `w`×`h` frame; it is cropped to even dimensions in place.
    fn frame(&mut self, rgb: &mut [u8], w: u32, h: u32, ms: u32) -> Result<(), SitiError> {
        if Instant::now() > self.deadline {
            return Err(SitiError::Unsupported("took too long".into()));
        }
        let (ew, eh) = (w & !1, h & !1);
        if self.state.is_none() {
            check_size(w, h)?;
            let fps = self.fps.unwrap_or(1000.0 / ms as f32);
            self.state = Some((Captions::new(self.top, self.bottom, ew, eh)?, Mp4Out::new(ew, eh, fps)?));
        }
        let (captions, out) = self.state.as_mut().expect("initialized above");
        if (out.w, out.h) != (ew, eh) {
            return Err(SitiError::Unsupported("frame size changed mid-stream".into()));
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
    let decoder = GifDecoder::new(Cursor::new(data)).map_err(|e| SitiError::Unsupported(format!("unreadable GIF: {e}")))?;
    let (w, h) = decoder.dimensions();
    check_size(w, h)?;
    let mut pipeline = Pipeline::new(top, bottom, None);
    let mut rgb = vec![0u8; w as usize * h as usize * 3];
    let mut count = 0;
    for frame in decoder.into_frames() {
        count += 1;
        check_frames(count)?;
        let frame = frame.map_err(|e| SitiError::Unsupported(format!("unreadable GIF: {e}")))?;
        let (n, d) = frame.delay().numer_denom_ms();
        let ms = n / d.max(1);
        let ms = if ms < MIN_GIF_DELAY_MS { DEFAULT_GIF_DELAY_MS } else { ms };
        let buf = frame.into_buffer();
        if buf.dimensions() != (w, h) {
            return Err(SitiError::Unsupported("GIF frame size differs from the canvas".into()));
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
    let input = Mp4Input::open(data)?;
    let mut pipeline = Pipeline::new(top, bottom, Some(input.fps));
    input.decode(|rgb, w, h, ms| pipeline.frame(rgb, w, h, ms))?;
    pipeline.finish()
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
            .ok_or_else(|| SitiError::Unsupported("corrupt H.264 sample".into()))?;
        let len = len.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize);
        let (nal, tail) = tail.split_at_checked(len).ok_or_else(|| SitiError::Unsupported("corrupt H.264 sample".into()))?;
        if nal.first().is_some_and(|b| b & 0x1F == 7) {
            check_sps(nal)?;
        }
        out.extend_from_slice(&START);
        out.extend_from_slice(nal);
        rest = tail;
    }
    Ok(())
}

/// MSB-first bit reader over an unescaped RBSP; every read past the end returns `None`.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn u(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            v = (v << 1) | ((byte >> (7 - self.pos % 8)) & 1) as u32;
            self.pos += 1;
        }
        Some(v)
    }

    fn ue(&mut self) -> Option<u32> {
        let mut lz = 0;
        while self.u(1)? == 0 {
            lz += 1;
            if lz > 31 {
                return None;
            }
        }
        Some((1u32 << lz) - 1 + self.u(lz)?)
    }

    fn se(&mut self) -> Option<i32> {
        let k = self.ue()? as i64;
        Some(if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) } as i32)
    }
}

/// Coded size (macroblock-aligned, before cropping) declared by an SPS NAL unit (header byte included).
fn sps_coded_size(sps: &[u8]) -> Option<(u32, u32)> {
    let mut rbsp = Vec::with_capacity(sps.len());
    let mut zeros = 0;
    for &b in sps.get(1..)? {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        rbsp.push(b);
    }
    let mut r = Bits { data: &rbsp, pos: 0 };
    let profile_idc = r.u(8)?;
    r.u(8)?;
    r.u(8)?;
    r.ue()?;
    if matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        let chroma = r.ue()?;
        if chroma == 3 {
            r.u(1)?;
        }
        r.ue()?;
        r.ue()?;
        r.u(1)?;
        if r.u(1)? == 1 {
            for i in 0..if chroma == 3 { 12 } else { 8 } {
                if r.u(1)? == 1 {
                    let (mut last, mut next) = (8i64, 8i64);
                    for _ in 0..if i < 6 { 16 } else { 64 } {
                        if next != 0 {
                            next = (last + r.se()? as i64 + 256).rem_euclid(256);
                        }
                        if next != 0 {
                            last = next;
                        }
                    }
                }
            }
        }
    }
    r.ue()?;
    match r.ue()? {
        0 => {
            r.ue()?;
        }
        1 => {
            r.u(1)?;
            r.se()?;
            r.se()?;
            let n = r.ue()?;
            if n > 255 {
                return None;
            }
            for _ in 0..n {
                r.se()?;
            }
        }
        _ => {}
    }
    r.ue()?;
    r.u(1)?;
    let w_mbs = r.ue()?.checked_add(1)?;
    let h_units = r.ue()?.checked_add(1)?;
    let frame_mbs_only = r.u(1)?;
    Some((w_mbs.checked_mul(16)?, h_units.checked_mul(16)?.checked_mul(2 - frame_mbs_only)?))
}

/// Rejects an SPS that is unreadable or declares a coded size over [`MAX_SIDE`].
fn check_sps(sps: &[u8]) -> Result<(), SitiError> {
    let (w, h) = sps_coded_size(sps).ok_or_else(|| SitiError::Unsupported("unreadable SPS".into()))?;
    let max = MAX_SIDE.next_multiple_of(16);
    if w > max || h > max {
        return Err(SitiError::Unsupported(format!("coded size {w}x{h} over {MAX_SIDE}")));
    }
    Ok(())
}

/// Parsed MP4 header plus everything needed to decode its first H.264 track.
struct Mp4Input<'a> {
    reader: Mp4Reader<Cursor<&'a [u8]>>,
    track_id: u32,
    sample_count: u32,
    timescale: u64,
    length_size: usize,
    /// SPS then PPS NAL units from avcC.
    params: Vec<Vec<u8>>,
    fps: f32,
}

impl<'a> Mp4Input<'a> {
    fn open(data: &'a [u8]) -> Result<Self, SitiError> {
        let reader = Mp4Reader::read_header(Cursor::new(data), data.len() as u64)
            .map_err(|e| SitiError::Unsupported(format!("unreadable MP4: {e}")))?;
        let track = h264_track(&reader)?;
        let sample_count = track.sample_count();
        check_frames(sample_count)?;
        check_size(track.width() as u32, track.height() as u32)?;
        let track_id = track.track_id();
        let timescale = track.timescale().max(1) as u64;
        let avcc = &track.trak.mdia.minf.stbl.stsd.avc1.as_ref().ok_or_else(|| internal("missing avc1 box"))?.avcc;
        let length_size = avcc.length_size_minus_one as usize + 1;
        let params: Vec<Vec<u8>> = avcc
            .sequence_parameter_sets
            .iter()
            .chain(&avcc.picture_parameter_sets)
            .map(|n| n.bytes.clone())
            .collect();
        for sps in params.iter().filter(|p| p.first().is_some_and(|b| b & 0x1F == 7)) {
            check_sps(sps)?;
        }
        let secs = track.duration().as_secs_f32();
        let fps = if secs > 0.0 { sample_count as f32 / secs } else { 30.0 };
        Ok(Self { reader, track_id, sample_count, timescale, length_size, params, fps })
    }

    /// Decodes every sample, calling `sink(rgb, w, h, duration_ms)` per frame in display order.
    fn decode(mut self, mut sink: impl FnMut(&mut [u8], u32, u32, u32) -> Result<(), SitiError>) -> Result<(), SitiError> {
        // NoFlush: flushing after every decode breaks B-frame reordering.
        let config = DecoderConfig::new().flush_after_decode(Flush::NoFlush);
        let mut decoder = Decoder::with_api_config(OpenH264API::from_source(), config).map_err(internal)?;
        let mut durations: Vec<u32> = Vec::with_capacity(self.sample_count as usize);
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
        for sample_id in 1..=self.sample_count {
            let Some(sample) = self
                .reader
                .read_sample(self.track_id, sample_id)
                .map_err(|e| SitiError::Unsupported(format!("unreadable MP4: {e}")))?
            else {
                continue;
            };
            durations.push(((sample.duration as u64 * 1000 / self.timescale) as u32).max(1));
            to_annex_b(&sample.bytes, self.length_size, sample.is_sync, &self.params, &mut annex_b)?;
            let decoded = decoder.decode(&annex_b).map_err(|e| SitiError::Unsupported(format!("corrupt H.264 sample {sample_id}: {e}")))?;
            if let Some(yuv) = decoded {
                emit(&yuv, &durations)?;
            }
        }
        let count = self.sample_count;
        let rest = decoder.flush_remaining().map_err(|e| SitiError::Unsupported(format!("corrupt H.264 sample {count}: {e}")))?;
        for yuv in rest {
            emit(&yuv, &durations)?;
        }
        Ok(())
    }
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

    /// `(sample_count, width, height, durations, sync flags)` of the only track.
    fn track_info(mp4: &[u8]) -> (u32, u32, u32, Vec<u32>, Vec<bool>) {
        let mut r = Mp4Reader::read_header(Cursor::new(mp4), mp4.len() as u64).unwrap();
        assert_eq!(r.tracks().len(), 1);
        let t = h264_track(&r).unwrap();
        let (id, n, w, h) = (t.track_id(), t.sample_count(), t.width() as u32, t.height() as u32);
        let (durs, syncs) = (1..=n)
            .map(|i| {
                let s = r.read_sample(id, i).unwrap().unwrap();
                (s.duration, s.is_sync)
            })
            .unzip();
        (n, w, h, durs, syncs)
    }

    #[test]
    fn mp4_round_trip_keeps_timing_and_draws_captions() {
        let out = add_text(&solid_mp4(320, 240, 70, 100), AnimationFormat::Mp4, "top text", "bottom text").unwrap();
        let (n, w, h, durs, syncs) = track_info(&out);
        assert_eq!((n, w, h), (70, 320, 240));
        assert!((durs.iter().sum::<u32>() as i64 - 7000).abs() <= 10);
        assert!(syncs[60], "sample 61 must be the second IDR");

        // Decoding across the second IDR proves the repeated SPS/PPS match the avcC ones.
        let (mut first, mut frames) = (None, 0);
        Mp4Input::open(&out)
            .unwrap()
            .decode(|rgb, _, _, _| {
                first.get_or_insert_with(|| rgb.to_vec());
                frames += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(frames, 70);
        let img = first.unwrap();
        let white = |y0: usize, y1: usize| (y0..y1).any(|y| (0..320).any(|x| img[(y * 320 + x) * 3..][..3].iter().all(|&c| c > 200)));
        assert!(white(0, 72), "top caption missing");
        assert!(white(168, 240), "bottom caption missing");
        assert!(!white(96, 144), "middle must stay untouched");
    }

    #[test]
    fn decodes_b_frames_in_display_order() {
        // x264 High profile, 3 B-frames, 30 gray frames with luma ≈ N*8.
        let mut centre = Vec::new();
        Mp4Input::open(include_bytes!("../assets/test/bframes.mp4"))
            .unwrap()
            .decode(|rgb, w, _, _| {
                centre.push(rgb[(32 * w as usize + 32) * 3]);
                Ok(())
            })
            .unwrap();
        assert_eq!(centre.len(), 30);
        assert!(centre.windows(2).all(|p| p[0] < p[1]), "frames out of display order: {centre:?}");
    }

    #[test]
    fn gif_odd_size_is_cropped_and_delays_kept() {
        let out = add_text(&gif(101, 75, 3, 70), AnimationFormat::Gif, "a", "b").unwrap();
        let (n, w, h, durs, _) = track_info(&out);
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
        let err = add_text(b"not a video", AnimationFormat::Mp4, "a", "").unwrap_err();
        assert!(matches!(err, SitiError::Unsupported(_)), "{err}");
        let err = add_text(b"not a gif", AnimationFormat::Gif, "a", "").unwrap_err();
        assert!(matches!(err, SitiError::Unsupported(_)), "{err}");
    }

    #[test]
    fn sps_size_matches_encoder_output() {
        let mp4 = solid_mp4(320, 240, 1, 100);
        assert_eq!(sps_coded_size(&Mp4Input::open(&mp4).unwrap().params[0]), Some((320, 240)));
        let fixture = include_bytes!("../assets/test/bframes.mp4");
        assert_eq!(sps_coded_size(&Mp4Input::open(fixture).unwrap().params[0]), Some((64, 64)));
    }

    #[test]
    fn sps_parse_rejects_truncated() {
        assert_eq!(sps_coded_size(&[0x67, 0x64]), None);
        assert!(matches!(check_sps(&[0x67, 0x64]), Err(SitiError::Unsupported(_))));
    }
}
