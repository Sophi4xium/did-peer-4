use sha2::{Sha256, Digest};
use serde_json::{json, Value};

fn encode_document(doc: &Value) -> String {
    // JSON文字列をUTF-8バイトに変換
    let json_str = serde_json::to_string(doc).unwrap();
    let bytes = json_str.as_bytes();
    
    // base58btcでエンコードしてzプレフィックスを付ける
    let encoded = bs58::encode(bytes).into_string();
    format!("z{}", encoded)
}

fn hash_document(encoded: &str) -> String {
    // encodedDocumentのバイト列をSHA-256ハッシュ
    let mut hasher = Sha256::new();
    hasher.update(encoded.as_bytes());
    let hash_bytes = hasher.finalize();
    
    // base58btcでエンコードしてzプレフィックスを付ける
    let encoded_hash = bs58::encode(hash_bytes).into_string();
    format!("z{}", encoded_hash)
}

fn create_did_peer_4(input_doc: &Value) -> (String, String) {
    let encoded_doc = encode_document(input_doc);
    let hash = hash_document(&encoded_doc);
    
    let long_form = format!("did:peer:4{}:{}", hash, encoded_doc);
    let short_form = format!("did:peer:4{}", hash);
    
    (short_form, long_form)
}

fn resolve(long_form_did: &str) -> Value {
    // "did:peer:4{hash}:{encodedDocument}" → split後
    let after_prefix = long_form_did.strip_prefix("did:peer:4").unwrap();
    let parts: Vec<&str> = after_prefix.splitn(2, ':').collect();
    
    let hash = parts[0];
    let encoded_doc = parts[1];
    
    // ハッシュ検証
    let expected_hash = hash_document(encoded_doc);
    assert_eq!(hash, expected_hash, "ハッシュが一致しません");
    
    // base58btcデコード（zプレフィックスを除去）
    let encoded_without_prefix = encoded_doc.strip_prefix('z').unwrap();
    let decoded_bytes = bs58::decode(encoded_without_prefix).into_vec().unwrap();
    let json_str = String::from_utf8(decoded_bytes).unwrap();
    
    let mut doc: Value = serde_json::from_str(&json_str).unwrap();
    
    // idとalsoKnownAsを追加
    let short_form = format!("did:peer:4{}", hash);
    doc["id"] = json!(long_form_did);
    doc["alsoKnownAs"] = json!([short_form]);
    
    doc
}

fn main() {
    // Input Document（idなし）
    let input_doc = json!({
        "@context": ["https://www.w3.org/ns/did/v1"],
        "authentication": [{
            "type": "Ed25519VerificationKey2020",
            "publicKeyMultibase": "z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH"
        }],
        "keyAgreement": [{
            "type": "X25519KeyAgreementKey2020",
            "publicKeyMultibase": "z6LSbysY2xFMRpGMhb7tFTLMpeuPRaqaWM1yECx2AtzE3KCc"
        }],
        "service": [{
            "type": "DIDCommMessaging",
            "serviceEndpoint": "http://localhost:8002/didcomm"
        }]
    });

    println!("=== did:peer:4 生成 ===\n");
    println!("Input Document:\n{}\n", serde_json::to_string_pretty(&input_doc).unwrap());

    let (short_form, long_form) = create_did_peer_4(&input_doc);

    println!("Short Form DID:\n{}\n", short_form);
    println!("Long Form DID（先頭100文字）:\n{}...\n", &long_form[..100]);

    println!("=== 解決（Resolve）===\n");
    let resolved = resolve(&long_form);
    println!("Resolved DID Document:\n{}", serde_json::to_string_pretty(&resolved).unwrap());
}