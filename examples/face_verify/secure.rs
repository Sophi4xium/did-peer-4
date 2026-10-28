//! 特徴ベクトルの暗号化(共通鍵)と文字列化。
//!
//! - 暗号方式 : AES-256-GCM(認証付き暗号。改ざんも検出できる)
//! - 鍵       : 32 バイト。URL-safe Base64(パディングなし)文字列で受け渡す
//! - 出力形式 : `fv1:` + Base64url( nonce(12B) || 暗号文+認証タグ )
//! - 平文形式 : version(1B) || 次元数(u32 LE) || f32 LE × 次元数
//!
//! nonce は暗号化のたびにランダム生成するので、同じベクトルでも毎回違う文字列になる。
//! (同じ鍵で nonce を使い回すと GCM は破綻するため、必ず毎回生成すること)

use std::error::Error;
use std::fmt;

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

use crate::models::{Embedding, FaceError, Verifier, VerifyResult};

const PREFIX: &str = "fv1:";
const VERSION: u8 = 1;
const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;
/// 追加認証データ。形式・バージョンが違うデータの復号を弾く。
const AAD: &[u8] = b"face_verify:v1";
/// 異常に大きい次元数を弾く上限(メモリ確保の暴走防止)
const MAX_DIM: usize = 65_536;

#[derive(Debug, Clone, PartialEq)]
pub enum CryptoError {
    /// 鍵の形式が不正(Base64 でない / 32 バイトでない)
    InvalidKey,
    /// 暗号化に失敗
    EncryptFailed,
    /// 復号に失敗(鍵違い・改ざん・形式不正のいずれか。理由は区別しない)
    DecryptFailed,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::InvalidKey => write!(f, "鍵が不正です(Base64 で 32 バイトが必要)"),
            CryptoError::EncryptFailed => write!(f, "暗号化に失敗しました"),
            CryptoError::DecryptFailed => {
                write!(f, "復号に失敗しました(鍵の違い・データの破損・形式不正)")
            }
        }
    }
}

impl Error for CryptoError {}

pub struct EmbeddingCipher {
    cipher: Aes256Gcm,
}

impl EmbeddingCipher {
    pub fn new(key: &[u8; KEY_LEN]) -> Self {
        Self {
            cipher: Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key)),
        }
    }

    /// Base64url 文字列の鍵から作る(環境変数などから受け取る想定)。
    pub fn from_base64_key(s: &str) -> Result<Self, CryptoError> {
        let bytes = URL_SAFE_NO_PAD
            .decode(s.trim())
            .map_err(|_| CryptoError::InvalidKey)?;
        let key: [u8; KEY_LEN] = bytes.try_into().map_err(|_| CryptoError::InvalidKey)?;
        Ok(Self::new(&key))
    }

    /// 新しいランダム鍵を生成して Base64url 文字列で返す。
    pub fn generate_key_base64() -> String {
        let key = Aes256Gcm::generate_key(&mut OsRng);
        URL_SAFE_NO_PAD.encode(key.as_slice())
    }

    /// 特徴ベクトルを暗号化して文字列にする。
    pub fn encrypt(&self, emb: &Embedding) -> Result<String, CryptoError> {
        let values = emb.as_slice();

        // 平文: version || dim(u32 LE) || f32 LE...
        let mut plain = Vec::with_capacity(1 + 4 + values.len() * 4);
        plain.push(VERSION);
        plain.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for v in values {
            plain.extend_from_slice(&v.to_le_bytes());
        }

        let nonce = Aes256Gcm::generate_nonce(&mut OsRng); // 毎回ランダム(12B)
        let ciphertext = self
            .cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: &plain,
                    aad: AAD,
                },
            )
            .map_err(|_| CryptoError::EncryptFailed)?;

        let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        blob.extend_from_slice(&nonce);
        blob.extend_from_slice(&ciphertext);
        Ok(format!("{PREFIX}{}", URL_SAFE_NO_PAD.encode(blob)))
    }

    /// `encrypt` で作った文字列から特徴ベクトルを復元する。
    pub fn decrypt(&self, token: &str) -> Result<Embedding, CryptoError> {
        let body = token
            .trim()
            .strip_prefix(PREFIX)
            .ok_or(CryptoError::DecryptFailed)?;
        let blob = URL_SAFE_NO_PAD
            .decode(body)
            .map_err(|_| CryptoError::DecryptFailed)?;
        // nonce + (最低でも認証タグ 16B)
        if blob.len() < NONCE_LEN + 16 {
            return Err(CryptoError::DecryptFailed);
        }

        let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
        let plain = self
            .cipher
            .decrypt(
                Nonce::from_slice(nonce_bytes),
                Payload {
                    msg: ciphertext,
                    aad: AAD,
                },
            )
            .map_err(|_| CryptoError::DecryptFailed)?;

        parse_plain(&plain)
    }
}

fn parse_plain(plain: &[u8]) -> Result<Embedding, CryptoError> {
    if plain.len() < 5 || plain[0] != VERSION {
        return Err(CryptoError::DecryptFailed);
    }
    let dim = u32::from_le_bytes([plain[1], plain[2], plain[3], plain[4]]) as usize;
    if dim == 0 || dim > MAX_DIM || plain.len() != 5 + dim * 4 {
        return Err(CryptoError::DecryptFailed);
    }

    let values: Vec<f32> = plain[5..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // 値の妥当性チェック(NaN/Inf/ゼロベクトル)と再正規化
    Embedding::new(values).map_err(|_| CryptoError::DecryptFailed)
}

/// 暗号化トークンとの照合で起こりうるエラー。
#[derive(Debug, Clone, PartialEq)]
pub enum TokenVerifyError {
    /// トークンの復号に失敗(鍵の違い・改ざん・形式不正)
    Crypto(CryptoError),
    /// 照合に失敗(次元数の不一致など)
    Face(FaceError),
}

impl fmt::Display for TokenVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenVerifyError::Crypto(e) => write!(f, "{e}"),
            TokenVerifyError::Face(e) => write!(f, "{e}"),
        }
    }
}

impl Error for TokenVerifyError {}

impl From<CryptoError> for TokenVerifyError {
    fn from(e: CryptoError) -> Self {
        TokenVerifyError::Crypto(e)
    }
}

impl From<FaceError> for TokenVerifyError {
    fn from(e: FaceError) -> Self {
        TokenVerifyError::Face(e)
    }
}

impl Verifier {
    /// 暗号化済みの登録データ(`EmbeddingCipher::encrypt` の出力)と、
    /// 新しく抽出した特徴ベクトル `probe` を照合する。
    ///
    /// 登録データはこの関数の中で復号し、照合後すぐに破棄される。
    pub fn verify_encrypted(
        &self,
        cipher: &EmbeddingCipher,
        probe: &Embedding,
        token: &str,
    ) -> Result<VerifyResult, TokenVerifyError> {
        let enrolled = cipher.decrypt(token)?;
        Ok(self.verify(&enrolled, probe)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Embedding {
        Embedding::new((0..512).map(|i| ((i as f32) * 0.37).sin()).collect()).unwrap()
    }

    fn new_cipher() -> EmbeddingCipher {
        EmbeddingCipher::from_base64_key(&EmbeddingCipher::generate_key_base64()).unwrap()
    }

    #[test]
    fn roundtrip_restores_same_vector() {
        let c = new_cipher();
        let original = sample();
        let token = c.encrypt(&original).unwrap();
        assert!(token.starts_with("fv1:"));

        let restored = c.decrypt(&token).unwrap();
        assert_eq!(restored.dim(), original.dim());
        let sim = original.cosine_similarity(&restored).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn same_vector_gives_different_tokens() {
        let c = new_cipher();
        let e = sample();
        assert_ne!(c.encrypt(&e).unwrap(), c.encrypt(&e).unwrap()); // nonce が毎回違う
    }

    #[test]
    fn token_is_plain_ascii_without_whitespace() {
        let token = new_cipher().encrypt(&sample()).unwrap();
        assert!(token.is_ascii() && !token.chars().any(|c| c.is_whitespace()));
    }

    #[test]
    fn wrong_key_fails() {
        let token = new_cipher().encrypt(&sample()).unwrap();
        assert_eq!(
            new_cipher().decrypt(&token).unwrap_err(),
            CryptoError::DecryptFailed
        );
    }

    #[test]
    fn tampering_is_detected() {
        let c = new_cipher();
        let token = c.encrypt(&sample()).unwrap();
        let mut blob = URL_SAFE_NO_PAD.decode(&token[PREFIX.len()..]).unwrap();
        let last = blob.len() - 1;
        blob[last] ^= 0x01; // 1ビットだけ改ざん
        let tampered = format!("{PREFIX}{}", URL_SAFE_NO_PAD.encode(blob));
        assert_eq!(
            c.decrypt(&tampered).unwrap_err(),
            CryptoError::DecryptFailed
        );
    }

    #[test]
    fn malformed_tokens_fail() {
        let c = new_cipher();
        for bad in ["", "fv1:", "fv1:!!!", "xx1:AAAA", "hello"] {
            assert_eq!(
                c.decrypt(bad).unwrap_err(),
                CryptoError::DecryptFailed,
                "{bad}"
            );
        }
    }

    #[test]
    fn invalid_keys_are_rejected() {
        assert!(EmbeddingCipher::from_base64_key("not base64 !!").is_err());
        assert!(EmbeddingCipher::from_base64_key(&URL_SAFE_NO_PAD.encode([0u8; 16])).is_err());
    }

    #[test]
    fn verify_encrypted_matches_same_person_only() {
        let c = new_cipher();
        let v = Verifier::new(0.5).unwrap();
        let enrolled = sample();
        let token = c.encrypt(&enrolled).unwrap();

        // 同じベクトル(=同一人物)→ 一致
        let same = v.verify_encrypted(&c, &enrolled, &token).unwrap();
        assert!(same.is_match && (same.score - 1.0).abs() < 1e-5);

        // 全く違うベクトル → 不一致
        let other = Embedding::new((0..512).map(|i| ((i as f32) * 1.91).cos()).collect()).unwrap();
        let diff = v.verify_encrypted(&c, &other, &token).unwrap();
        assert!(!diff.is_match);
    }

    #[test]
    fn verify_encrypted_reports_crypto_error() {
        let v = Verifier::new(0.5).unwrap();
        let token = new_cipher().encrypt(&sample()).unwrap();
        let err = v
            .verify_encrypted(&new_cipher(), &sample(), &token)
            .unwrap_err();
        assert_eq!(err, TokenVerifyError::Crypto(CryptoError::DecryptFailed));
    }

    #[test]
    fn verify_encrypted_reports_dimension_mismatch() {
        let c = new_cipher();
        let v = Verifier::new(0.5).unwrap();
        let token = c.encrypt(&sample()).unwrap(); // 512次元
        let short = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        let err = v.verify_encrypted(&c, &short, &token).unwrap_err();
        assert_eq!(
            err,
            TokenVerifyError::Face(FaceError::DimensionMismatch {
                left: 512,
                right: 3
            })
        );
    }
}
