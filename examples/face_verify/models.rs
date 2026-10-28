//! 顔特徴ベクトル(embedding)をコサイン類似度で比較して同一人物かを判定するサンプル。
//! 特徴ベクトルの比較ロジック(外部クレート不要)。画像からの抽出は `encoder` モジュール。

use std::error::Error;
use std::fmt;

// ---------------------------------------------------------------------------
// エラー型
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum FaceError {
    EmptyVector,
    InvalidValue,
    ZeroNorm,
    DimensionMismatch { left: usize, right: usize },
    InvalidThreshold(f32),
}

impl fmt::Display for FaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FaceError::EmptyVector => write!(f, "特徴ベクトルが空です"),
            FaceError::InvalidValue => write!(f, "NaN または無限大が含まれています"),
            FaceError::ZeroNorm => write!(f, "ゼロベクトルは扱えません"),
            FaceError::DimensionMismatch { left, right } => {
                write!(f, "次元数が一致しません: {left} vs {right}")
            }
            FaceError::InvalidThreshold(t) => {
                write!(f, "閾値は -1.0..=1.0 の範囲で指定してください: {t}")
            }
        }
    }
}

impl Error for FaceError {}

// ---------------------------------------------------------------------------
// 特徴ベクトル(生成時に L2 正規化して保持する)
// ---------------------------------------------------------------------------

/// L2 正規化済みの顔特徴ベクトル。
/// 正規化済みなので、コサイン類似度は単なる内積で計算できる。
#[derive(Debug, Clone)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    pub fn new(raw: Vec<f32>) -> Result<Self, FaceError> {
        if raw.is_empty() {
            return Err(FaceError::EmptyVector);
        }
        if raw.iter().any(|v| !v.is_finite()) {
            return Err(FaceError::InvalidValue);
        }

        // 精度のため f64 で累積
        let norm = raw
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        if norm < 1e-12 {
            return Err(FaceError::ZeroNorm);
        }

        let normalized = raw.iter().map(|&v| (v as f64 / norm) as f32).collect();
        Ok(Self(normalized))
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// L2 正規化済みの値を読み取り専用で参照する(表示・保存用)。
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// コサイン類似度(-1.0 ~ 1.0)。正規化済みなので内積と等しい。
    pub fn cosine_similarity(&self, other: &Embedding) -> Result<f32, FaceError> {
        if self.dim() != other.dim() {
            return Err(FaceError::DimensionMismatch {
                left: self.dim(),
                right: other.dim(),
            });
        }
        let dot: f64 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        // 浮動小数点誤差で範囲外に出るのを防ぐ
        Ok((dot as f32).clamp(-1.0, 1.0))
    }
}

// ---------------------------------------------------------------------------
// 1:1 照合
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct VerifyResult {
    pub score: f32,
    pub is_match: bool,
}

pub struct Verifier {
    threshold: f32,
}

impl Verifier {
    pub fn new(threshold: f32) -> Result<Self, FaceError> {
        if !(-1.0..=1.0).contains(&threshold) {
            return Err(FaceError::InvalidThreshold(threshold));
        }
        Ok(Self { threshold })
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// 2つの特徴ベクトルが同一人物かを判定する。
    pub fn verify(&self, a: &Embedding, b: &Embedding) -> Result<VerifyResult, FaceError> {
        let score = a.cosine_similarity(b)?;
        Ok(VerifyResult {
            score,
            is_match: score >= self.threshold,
        })
    }
}

// ---------------------------------------------------------------------------
// 1:N 識別(登録済みユーザーの中から最も近い人物を探す)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Gallery {
    entries: Vec<(String, Embedding)>,
}

impl Gallery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enroll(&mut self, id: impl Into<String>, emb: Embedding) {
        self.entries.push((id.into(), emb));
    }

    /// 最もスコアが高い登録者を返す。閾値未満なら None(= 未登録の人物)。
    pub fn identify(
        &self,
        probe: &Embedding,
        verifier: &Verifier,
    ) -> Result<Option<(&str, f32)>, FaceError> {
        let mut best: Option<(&str, f32)> = None;
        for (id, emb) in &self.entries {
            let score = probe.cosine_similarity(emb)?;
            if best.map_or(true, |(_, s)| score > s) {
                best = Some((id.as_str(), score));
            }
        }
        Ok(best.filter(|&(_, score)| score >= verifier.threshold()))
    }
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let a = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        let b = Embedding::new(vec![2.0, 4.0, 6.0]).unwrap(); // 同じ向き
        let s = a.cosine_similarity(&b).unwrap();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![0.0, 1.0]).unwrap();
        assert!(a.cosine_similarity(&b).unwrap().abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_have_similarity_minus_one() {
        let a = Embedding::new(vec![1.0, 1.0]).unwrap();
        let b = Embedding::new(vec![-1.0, -1.0]).unwrap();
        assert!((a.cosine_similarity(&b).unwrap() + 1.0).abs() < 1e-5);
    }

    #[test]
    fn dimension_mismatch_is_error() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            a.cosine_similarity(&b),
            Err(FaceError::DimensionMismatch { left: 2, right: 3 })
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert_eq!(Embedding::new(vec![]).unwrap_err(), FaceError::EmptyVector);
        assert_eq!(
            Embedding::new(vec![0.0, 0.0]).unwrap_err(),
            FaceError::ZeroNorm
        );
        assert_eq!(
            Embedding::new(vec![f32::NAN, 1.0]).unwrap_err(),
            FaceError::InvalidValue
        );
    }

    #[test]
    fn threshold_decides_match() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![1.0, 1.0]).unwrap(); // cos ≈ 0.7071
        assert!(Verifier::new(0.7).unwrap().verify(&a, &b).unwrap().is_match);
        assert!(!Verifier::new(0.8).unwrap().verify(&a, &b).unwrap().is_match);
    }

    #[test]
    fn gallery_returns_best_or_none() {
        let v = Verifier::new(0.9).unwrap();
        let mut g = Gallery::new();
        g.enroll("x", Embedding::new(vec![1.0, 0.0]).unwrap());
        g.enroll("y", Embedding::new(vec![0.0, 1.0]).unwrap());

        let probe = Embedding::new(vec![0.99, 0.05]).unwrap();
        assert_eq!(g.identify(&probe, &v).unwrap().map(|(id, _)| id), Some("x"));

        let unknown = Embedding::new(vec![1.0, 1.0]).unwrap(); // どちらとも ≈0.707
        assert!(g.identify(&unknown, &v).unwrap().is_none());
    }
}
