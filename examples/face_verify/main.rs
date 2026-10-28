//! 使い方:
//!   face_verify <model.onnx> <画像A> <画像B> [閾値] [--show] [--full] [--encrypt]
//!   face_verify verify-token <model.onnx> <画像> <トークン> [閾値]
//!   face_verify gen-key
//!   face_verify decrypt <トークン> [--show] [--full]
//!
//! サブコマンド:
//!   (なし)        2枚の画像を比較する
//!   verify-token  画像と、暗号化済みの登録データ(fv1:...)を照合する
//!   gen-key       暗号化用の共通鍵を生成して表示
//!   decrypt       暗号化文字列を復号して表示
//!
//! オプション:
//!   --show     特徴ベクトルの先頭8要素と統計(最小/最大/平均/L2ノルム)を表示
//!   --full     特徴ベクトルの全要素を JSON 配列で表示
//!   --encrypt  特徴ベクトルを AES-256-GCM で暗号化し、文字列(fv1:...)で表示
//!
//! トークンは直接指定するほか、`@ファイル名` でファイルから読み込める
//! (約2,800文字と長いため)。
//!
//! 鍵は環境変数 FACE_VERIFY_KEY で渡す(コマンドライン引数に鍵を書くと履歴に残るため)。
//!
//! 終了コード:
//!   0 … 成功(verify-token では「同一人物」と判定)
//!   1 … エラー(鍵の不備、復号失敗、画像の読み込み失敗など)
//!   2 … 使い方が誤っている
//!   3 … verify-token で「別人」と判定
//!
//! 例(登録 → 認証):
//!   export FACE_VERIFY_KEY=$(face_verify gen-key)
//!   face_verify models/buffalo_l/w600k_r50.onnx samples/obama.png samples/obama2.png --encrypt
//!       # 表示された [A] encrypted: の文字列を enrolled.txt に保存しておく
//!   face_verify verify-token models/buffalo_l/w600k_r50.onnx samples/obama2.png @enrolled.txt

use std::env;
use std::error::Error;
use std::fs;

use crate::encoder::FaceEncoder;
use crate::models::*;
use crate::secure::EmbeddingCipher;

mod encoder;
mod models;
mod secure;

/// 先頭に表示する要素数(--show)
const HEAD_LEN: usize = 8;
/// 共通鍵を渡す環境変数名
const KEY_ENV: &str = "FACE_VERIFY_KEY";
/// 閾値の既定値(モデル依存。必ず自分のデータで調整すること)
const DEFAULT_THRESHOLD: f32 = 0.4;

const EXIT_NO_MATCH: i32 = 3;

const USAGE: &str = "usage:
  face_verify <model.onnx> <image_a> <image_b> [threshold] [--show] [--full] [--encrypt]
  face_verify verify-token <model.onnx> <image> <token|@file> [threshold]
  face_verify gen-key
  face_verify decrypt <token|@file> [--show] [--full]";

fn main() {
    // Result の Debug 表示ではなく、読みやすいエラーメッセージを出す
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("error: {e:?}");
            std::process::exit(1);
        }
    }
}

/// 終了コードを返す。
fn run() -> Result<i32, Box<dyn Error>> {
    let raw: Vec<String> = env::args().skip(1).collect();

    // サブコマンド
    match raw.first().map(String::as_str) {
        Some("gen-key") => {
            println!("{}", EmbeddingCipher::generate_key_base64());
            return Ok(0);
        }
        Some("decrypt") => return run_decrypt(&raw[1..]),
        Some("verify-token") => return run_verify_token(&raw[1..]),
        _ => {}
    }
    run_compare(raw)
}

/// 2枚の画像を比較する(従来の機能)。
fn run_compare(raw: Vec<String>) -> Result<i32, Box<dyn Error>> {
    // "--" で始まるものをフラグ、それ以外を位置引数として分ける
    let (flags, positional) = split_args(raw);
    check_flags(&flags, &["--show", "--full", "--encrypt"]);
    let show = has(&flags, "--show");
    let full = has(&flags, "--full");
    let encrypt = has(&flags, "--encrypt");

    if positional.len() < 3 {
        usage_exit();
    }
    let threshold = parse_threshold(positional.get(3))?;

    // 鍵の不備は、重いモデル読み込みの前に検出する
    let cipher = if encrypt { Some(load_cipher()?) } else { None };

    // モデルは1回だけ読み込んで使い回す
    let mut encoder = FaceEncoder::new(&positional[0])?;
    let emb_a = encoder.encode_path(&positional[1])?;
    let emb_b = encoder.encode_path(&positional[2])?;

    if show {
        print_summary("A", &positional[1], &emb_a);
        print_summary("B", &positional[2], &emb_b);
    }
    if full {
        print_full("A", &emb_a);
        print_full("B", &emb_b);
    }
    if let Some(c) = &cipher {
        println!("[A] encrypted: {}", c.encrypt(&emb_a)?);
        println!("[B] encrypted: {}", c.encrypt(&emb_b)?);
    }

    let verifier = Verifier::new(threshold)?;
    let result = verifier.verify(&emb_a, &emb_b)?;

    println!("dim       : {}", emb_a.dim());
    print_result(result.score, verifier.threshold(), result.is_match);
    Ok(0)
}

/// `verify-token <model> <image> <token|@file> [threshold]`:
/// 画像から特徴ベクトルを抽出し、暗号化済みの登録データと照合する。
fn run_verify_token(args: &[String]) -> Result<i32, Box<dyn Error>> {
    let (flags, positional) = split_args(args.to_vec());
    check_flags(&flags, &[]);
    if positional.len() < 3 {
        usage_exit();
    }
    let threshold = parse_threshold(positional.get(3))?;

    // 入力の不備(鍵・トークン)は、重いモデル読み込みの前に検出する
    let cipher = load_cipher()?;
    let token = read_token(&positional[2])?;
    let verifier = Verifier::new(threshold)?;

    let mut encoder = FaceEncoder::new(&positional[0])?;
    let probe = encoder.encode_path(&positional[1])?;

    // 登録データの復号は照合の直前に、この中でだけ行われる
    let result = verifier.verify_encrypted(&cipher, &probe, &token)?;

    print_result(result.score, verifier.threshold(), result.is_match);
    Ok(if result.is_match { 0 } else { EXIT_NO_MATCH })
}

/// `decrypt <token|@file>`: 暗号化文字列を復号して表示する。
fn run_decrypt(args: &[String]) -> Result<i32, Box<dyn Error>> {
    let (flags, positional) = split_args(args.to_vec());
    check_flags(&flags, &["--show", "--full"]);
    let full = has(&flags, "--full");

    let Some(arg) = positional.first() else {
        usage_exit();
    };

    let emb = load_cipher()?.decrypt(&read_token(arg)?)?;
    print_summary("decrypted", "(token)", &emb);
    if full {
        print_full("decrypted", &emb);
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// 補助関数
// ---------------------------------------------------------------------------

/// 環境変数から共通鍵を読み込む。
fn load_cipher() -> Result<EmbeddingCipher, Box<dyn Error>> {
    let key = env::var(KEY_ENV).map_err(|_| {
        format!("環境変数 {KEY_ENV} が未設定です(`face_verify gen-key` で生成できます)")
    })?;
    Ok(EmbeddingCipher::from_base64_key(&key)?)
}

/// トークン引数を読む。`@path` ならファイルから、それ以外は文字列そのもの。
fn read_token(arg: &str) -> Result<String, Box<dyn Error>> {
    match arg.strip_prefix('@') {
        Some(path) => {
            let text = fs::read_to_string(path)
                .map_err(|e| format!("トークンファイルを読み込めません ({path}): {e}"))?;
            Ok(text.trim().to_string())
        }
        None => Ok(arg.trim().to_string()),
    }
}

fn parse_threshold(arg: Option<&String>) -> Result<f32, Box<dyn Error>> {
    match arg {
        Some(s) => Ok(s
            .parse()
            .map_err(|_| format!("閾値を数値として解釈できません: {s}"))?),
        None => Ok(DEFAULT_THRESHOLD),
    }
}

fn print_result(score: f32, threshold: f32, is_match: bool) {
    println!("similarity: {score:.4}");
    println!("threshold : {threshold:.2}");
    println!("result    : {}", if is_match { "同一人物" } else { "別人" });
}

fn split_args(args: Vec<String>) -> (Vec<String>, Vec<String>) {
    args.into_iter().partition(|a| a.starts_with("--"))
}

fn has(flags: &[String], name: &str) -> bool {
    flags.iter().any(|f| f == name)
}

fn check_flags(flags: &[String], allowed: &[&str]) {
    for f in flags {
        if !allowed.contains(&f.as_str()) {
            eprintln!("unknown option: {f}\n{USAGE}");
            std::process::exit(2);
        }
    }
}

fn usage_exit() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2);
}

/// 先頭の数要素と統計値を表示する。
fn print_summary(label: &str, path: &str, emb: &Embedding) {
    let v = emb.as_slice();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
    // 正規化済みなので、ほぼ 1.0 になるはず(確認用)
    let l2 = v
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();

    let head: Vec<String> = v
        .iter()
        .take(HEAD_LEN)
        .map(|x| format!("{x:+.4}"))
        .collect();
    println!("[{label}] {path}");
    println!("  dim   : {}", v.len());
    println!("  head  : [{}, ...]", head.join(", "));
    println!("  min/max/mean: {min:+.4} / {max:+.4} / {mean:+.6}");
    println!("  L2 norm     : {l2:.6}");
}

/// 全要素を JSON 配列として1行で表示する。
fn print_full(label: &str, emb: &Embedding) {
    let items: Vec<String> = emb.as_slice().iter().map(|x| format!("{x:.6}")).collect();
    println!("[{label}] [{}]", items.join(", "));
}
