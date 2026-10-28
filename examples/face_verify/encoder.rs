//! 顔画像 → 特徴ベクトル(embedding)の抽出。
//!
//! InsightFace の ArcFace 系 ONNX モデル(例: buffalo_l 同梱の `w600k_r50.onnx`)を想定。
//! - 入力 : [1, 3, 112, 112] / float32 / RGB / (pixel - 127.5) / 127.5
//! - 出力 : [1, 512] の特徴ベクトル
//!
//! 注意: 入力は「顔検出 + 5点アライメント済みの 112x112 顔画像」が前提。
//! ここでは検出・アライメントは行わず、単純に 112x112 へリサイズするだけ。
//! (アライメントなしだと精度はかなり落ちます)

use std::error::Error;
use std::path::Path;

use image::{DynamicImage, imageops::FilterType};
use ort::{inputs, session::Session, value::Tensor};

use crate::models::Embedding;

/// モデルの入力解像度(ArcFace 系は 112x112)
pub const INPUT_SIZE: usize = 112;

pub struct FaceEncoder {
    session: Session,
}

impl FaceEncoder {
    /// ONNX モデルを読み込む。
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let env = ort::init().build()?;
        let session = Session::builder(&env)?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// 画像ファイルから特徴ベクトルを抽出する。
    pub fn encode_path(&mut self, path: impl AsRef<Path>) -> Result<Embedding, Box<dyn Error>> {
        let path = path.as_ref();
        let img = image::open(path)
            .map_err(|e| format!("画像を読み込めません ({}): {e}", path.display()))?;
        self.encode(&img)
    }

    /// デコード済み画像から特徴ベクトルを抽出する。
    pub fn encode(&mut self, img: &DynamicImage) -> Result<Embedding, Box<dyn Error>> {
        let input = preprocess(img);
        let tensor = Tensor::from_array((
            [1usize, 3, INPUT_SIZE, INPUT_SIZE],
            input.into_boxed_slice(),
        ))?;

        // ort 2.x では Session::run は &mut self
        let outputs = self.session.run(inputs![tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // Embedding::new が L2 正規化・不正値チェックを行う
        Ok(Embedding::new(data.to_vec())?)
    }
}

/// 前処理: リサイズ → RGB → CHW(チャンネル先頭)→ (x - 127.5) / 127.5
pub fn preprocess(img: &DynamicImage) -> Vec<f32> {
    let size = INPUT_SIZE as u32;
    let rgb = img.resize_exact(size, size, FilterType::Triangle).to_rgb8();

    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut out = vec![0.0f32; 3 * plane];
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let idx = y as usize * INPUT_SIZE + x as usize;
        for c in 0..3 {
            out[c * plane + idx] = (pixel[c] as f32 - 127.5) / 127.5;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn preprocess_has_expected_len_and_range() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(200, 150, Rgb([255, 255, 255])));
        let v = preprocess(&img);
        assert_eq!(v.len(), 3 * INPUT_SIZE * INPUT_SIZE);
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-5)); // 白 → 1.0
    }

    #[test]
    fn preprocess_is_channel_first_rgb() {
        // 赤一色: R面が +1.0、G/B面が -1.0 になる
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(112, 112, Rgb([255, 0, 0])));
        let v = preprocess(&img);
        let plane = INPUT_SIZE * INPUT_SIZE;
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[plane] + 1.0).abs() < 1e-5);
        assert!((v[2 * plane] + 1.0).abs() < 1e-5);
    }
}
