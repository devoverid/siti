//! Classic meme captions: white uppercase text with a black outline drawn at the top and/or
//! bottom of an existing image. Pure rasterization, no image generation model involved.

use std::{io::Cursor, sync::LazyLock};

use ab_glyph::{point, Font, FontRef, GlyphId, PxScale, ScaleFont};
use image::{codecs::jpeg::JpegEncoder, RgbImage};

use crate::error::SitiError;

/// Anton (SIL Open Font License, see `assets/fonts/OFL.txt`), an Impact-style display face.
static FONT: LazyLock<FontRef<'static>> = LazyLock::new(|| {
    FontRef::try_from_slice(include_bytes!("../assets/fonts/Anton-Regular.ttf")).expect("bundled font is valid")
});

/// Longest accepted caption per side, in characters.
pub const MAX_TEXT_CHARS: usize = 200;
const JPEG_QUALITY: u8 = 90;
/// Horizontal room a line may use, as a fraction of the image width.
const MAX_WIDTH_RATIO: f32 = 0.94;
/// Vertical room one caption block may use, as a fraction of the image height.
const MAX_BLOCK_HEIGHT_RATIO: f32 = 0.3;
const MARGIN_RATIO: f32 = 0.03;
const MIN_FONT_PX: f32 = 12.0;
/// Outline directions; 16 samples on the circle give a smooth stroke at meme stroke widths.
const STROKE_SAMPLES: usize = 16;

/// Draws `top` and `bottom` (either may be empty) onto `image` and returns the result as JPEG.
/// CPU-bound: call it from a blocking context, or use [`add_text_async`].
pub fn add_text(image: &[u8], top: &str, bottom: &str) -> Result<Vec<u8>, SitiError> {
    if top.chars().count() > MAX_TEXT_CHARS || bottom.chars().count() > MAX_TEXT_CHARS {
        return Err(SitiError::Generate(format!("caption is longer than {MAX_TEXT_CHARS} characters")));
    }
    let top = prepare(top);
    let bottom = prepare(bottom);
    if top.is_empty() && bottom.is_empty() {
        return Err(SitiError::Generate("caption text is empty".into()));
    }
    let mut img = image::load_from_memory(image)
        .map_err(|e| SitiError::Generate(format!("cannot read the image: {e}")))?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let margin = (h as f32 * MARGIN_RATIO).round();
    let mut canvas = Canvas::new(w, h);
    if !top.is_empty() {
        let block = Block::fit(&top, w as f32, h as f32);
        canvas.draw(&block, margin);
    }
    if !bottom.is_empty() {
        let block = Block::fit(&bottom, w as f32, h as f32);
        canvas.draw(&block, h as f32 - margin - block.height());
    }
    canvas.composite(&mut img);

    let mut out = Cursor::new(Vec::new());
    img.write_with_encoder(JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY))
        .map_err(|e| SitiError::Generate(format!("cannot encode the image: {e}")))?;
    Ok(out.into_inner())
}

/// [`add_text`] on the blocking thread pool.
pub async fn add_text_async(image: Vec<u8>, top: String, bottom: String) -> Result<Vec<u8>, SitiError> {
    tokio::task::spawn_blocking(move || add_text(&image, &top, &bottom))
        .await
        .map_err(|e| SitiError::Generate(format!("caption rendering failed: {e}")))?
}

/// Uppercases (meme convention), drops characters the font can't draw (emoji would render as
/// boxes), and collapses whitespace.
fn prepare(text: &str) -> String {
    let font = &*FONT;
    let mut out = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        let word: String = word.chars().flat_map(char::to_uppercase).filter(|&c| font.glyph_id(c) != GlyphId(0)).collect();
        if word.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&word);
    }
    out
}

/// A caption wrapped into lines at the largest font size that fits.
struct Block {
    scale: PxScale,
    lines: Vec<String>,
}

impl Block {
    fn fit(text: &str, img_w: f32, img_h: f32) -> Self {
        let max_w = img_w * MAX_WIDTH_RATIO;
        let max_h = img_h * MAX_BLOCK_HEIGHT_RATIO;
        let mut px = (img_h / 7.0).min(img_w / 5.0).max(MIN_FONT_PX);
        loop {
            let scale = PxScale::from(px);
            let lines = wrap(text, scale, max_w);
            let block = Block { scale, lines };
            let fits = block.height() <= max_h && block.lines.iter().all(|l| line_width(l, scale) <= max_w);
            if fits || px <= MIN_FONT_PX {
                return block;
            }
            px = (px * 0.9).max(MIN_FONT_PX);
        }
    }

    fn line_height(&self) -> f32 {
        FONT.as_scaled(self.scale).height()
    }

    fn height(&self) -> f32 {
        self.line_height() * self.lines.len() as f32
    }

    fn stroke_radius(&self) -> f32 {
        (self.scale.y / 16.0).max(1.0)
    }
}

/// Greedy word wrap; a single word wider than `max_w` keeps its own line and makes `fit` shrink.
fn wrap(text: &str, scale: PxScale, max_w: f32) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split(' ') {
        if current.is_empty() {
            current.push_str(word);
            continue;
        }
        let candidate_len = current.len() + 1 + word.len();
        current.push(' ');
        current.push_str(word);
        if line_width(&current, scale) > max_w {
            current.truncate(candidate_len - word.len() - 1);
            lines.push(std::mem::replace(&mut current, word.to_owned()));
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn line_width(line: &str, scale: PxScale) -> f32 {
    let font = FONT.as_scaled(scale);
    let mut width = 0.0;
    let mut prev = None;
    for c in line.chars() {
        let id = font.glyph_id(c);
        if let Some(prev) = prev {
            width += font.kern(prev, id);
        }
        width += font.h_advance(id);
        prev = Some(id);
    }
    width
}

/// Text coverage (fill) and outline coverage, composited onto the image in one pass.
struct Canvas {
    w: u32,
    h: u32,
    fill: Vec<f32>,
    stroke: Vec<f32>,
}

impl Canvas {
    fn new(w: u32, h: u32) -> Self {
        let len = w as usize * h as usize;
        Self { w, h, fill: vec![0.0; len], stroke: vec![0.0; len] }
    }

    /// Draws `block` centered horizontally with its top edge at `top`.
    fn draw(&mut self, block: &Block, top: f32) {
        let font = FONT.as_scaled(block.scale);
        let radius = block.stroke_radius();
        let offsets: Vec<(f32, f32)> = (0..STROKE_SAMPLES)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::TAU / STROKE_SAMPLES as f32;
                (angle.cos() * radius, angle.sin() * radius)
            })
            .chain(std::iter::once((0.0, 0.0)))
            .collect();
        for (i, line) in block.lines.iter().enumerate() {
            let baseline = top + block.line_height() * i as f32 + font.ascent();
            let mut x = (self.w as f32 - line_width(line, block.scale)) / 2.0;
            let mut prev = None;
            for c in line.chars() {
                let id = font.glyph_id(c);
                if let Some(prev) = prev {
                    x += font.kern(prev, id);
                }
                for &(dx, dy) in &offsets {
                    let is_fill = dx == 0.0 && dy == 0.0;
                    self.rasterize(id, block.scale, x + dx, baseline + dy, is_fill);
                }
                x += font.h_advance(id);
                prev = Some(id);
            }
        }
    }

    /// The outline is the glyph drawn at offsets around the circle; the fill doubles as stroke
    /// so thin interiors stay covered.
    fn rasterize(&mut self, id: GlyphId, scale: PxScale, x: f32, baseline: f32, fill: bool) {
        let Some(outlined) = FONT.outline_glyph(id.with_scale_and_position(scale, point(x, baseline))) else {
            return;
        };
        let bounds = outlined.px_bounds();
        let (w, h) = (self.w as i64, self.h as i64);
        outlined.draw(|gx, gy, coverage| {
            let px = bounds.min.x as i64 + gx as i64;
            let py = bounds.min.y as i64 + gy as i64;
            if px < 0 || py < 0 || px >= w || py >= h {
                return;
            }
            let idx = (py * w + px) as usize;
            self.stroke[idx] = self.stroke[idx].max(coverage);
            if fill {
                self.fill[idx] = self.fill[idx].max(coverage);
            }
        });
    }

    fn composite(&self, img: &mut RgbImage) {
        for ((pixel, &stroke), &fill) in img.pixels_mut().zip(&self.stroke).zip(&self.fill) {
            if stroke == 0.0 {
                continue;
            }
            for channel in &mut pixel.0 {
                let outlined = *channel as f32 * (1.0 - stroke.min(1.0));
                *channel = (outlined + (255.0 - outlined) * fill.min(1.0)).round() as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(w: u32, h: u32) -> Vec<u8> {
        let img = RgbImage::from_pixel(w, h, image::Rgb([40, 120, 200]));
        let mut out = Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn keeps_dimensions_and_draws_both_captions() {
        let out = add_text(&test_image(400, 300), "top text", "bottom text").unwrap();
        let img = image::load_from_memory(&out).unwrap().into_rgb8();
        assert_eq!(img.dimensions(), (400, 300));
        let white = |y0: u32, y1: u32| {
            (y0..y1).any(|y| (0..400).any(|x| img.get_pixel(x, y).0.iter().all(|&c| c > 230)))
        };
        assert!(white(0, 90), "top caption missing");
        assert!(white(210, 300), "bottom caption missing");
        assert!(!white(110, 190), "middle must stay untouched");
    }

    #[test]
    fn long_caption_stays_inside_the_image_width() {
        let text = "this is a very long caption that must wrap onto several lines instead of overflowing the image";
        let block = Block::fit(&prepare(text), 300.0, 300.0);
        assert!(block.lines.len() > 1);
        assert!(block.lines.iter().all(|l| line_width(l, block.scale) <= 300.0 * MAX_WIDTH_RATIO));
        assert!(block.height() <= 300.0 * MAX_BLOCK_HEIGHT_RATIO);
    }

    #[test]
    fn rejects_text_the_font_cannot_draw() {
        assert_eq!(prepare("hi 😂 there"), "HI THERE");
        assert!(add_text(&test_image(100, 100), "😂", "").is_err());
    }
}
