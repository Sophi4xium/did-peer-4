use sha2::{Sha256, Digest};
use serde_json::{json, Value};
use ed25519_dalek::SigningKey;
use x25519_dalek::{EphemeralSecret, PublicKey as X25519PublicKey};
use rand::RngExt;

fn multibase_encode(bytes: &[u8]) -> String {
    format!("z{}", bs58::encode(bytes).into_string())
}

fn multibase_decode(encoded: &str) -> Vec<u8> {
    let without_prefix = encoded.strip_prefix('z').unwrap();
    bs58::decode(without_prefix).into_vec().unwrap()
}

fn generate_ed25519_keypair() -> (SigningKey, String) {
    let mut rng = rand::rng();
    let secret_bytes: [u8; 32] = rng.random();
    let signing_key = SigningKey::from_bytes(&secret_bytes);
    let public_key_bytes = signing_key.verifying_key().to_bytes();
    let public_key_multibase = multibase_encode(&public_key_bytes);
    (signing_key, public_key_multibase)
}

fn generate_x25519_keypair() -> (EphemeralSecret, String) {
    let secret = EphemeralSecret::random_from_rng(&mut rand::rng());
    let public_key = X25519PublicKey::from(&secret);
    let public_key_multibase = multibase_encode(public_key.as_bytes());
    (secret, public_key_multibase)
}

fn encode_document(doc: &Value) -> String {
    let json_str = serde_json::to_string(doc).unwrap();
    multibase_encode(json_str.as_bytes())
}

fn hash_document(encoded: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(encoded.as_bytes());
    let hash_bytes = hasher.finalize();
    multibase_encode(&hash_bytes)
}

fn create_did_peer_4(input_doc: &Value) -> (String, String) {
    let encoded_doc = encode_document(input_doc);
    let hash = hash_document(&encoded_doc);
    let long_form = format!("did:peer:4{}:{}", hash, encoded_doc);
    let short_form = format!("did:peer:4{}", hash);
    (short_form, long_form)
}

fn resolve(long_form_did: &str) -> Value {
    let after_prefix = long_form_did.strip_prefix("did:peer:4").unwrap();
    let parts: Vec<&str> = after_prefix.splitn(2, ':').collect();
    let hash = parts[0];
    let encoded_doc = parts[1];

    let expected_hash = hash_document(encoded_doc);
    assert_eq!(hash, expected_hash, "ハッシュが一致しません");

    let decoded_bytes = multibase_decode(encoded_doc);
    let json_str = String::from_utf8(decoded_bytes).unwrap();
    let mut doc: Value = serde_json::from_str(&json_str).unwrap();

    let short_form = format!("did:peer:4{}", hash);
    doc["id"] = json!(long_form_did);
    doc["alsoKnownAs"] = json!([short_form]);

    doc
}

fn main() {
    println!("=== 鍵ペア生成 ===\n");

    let (_ed25519_signing_key, ed25519_public_multibase) = generate_ed25519_keypair();
    println!("Ed25519 公開鍵（multibase）: {}", ed25519_public_multibase);

    let (_x25519_secret, x25519_public_multibase) = generate_x25519_keypair();
    println!("X25519  公開鍵（multibase）: {}\n", x25519_public_multibase);

    let input_doc = json!({
        "@context": ["https://www.w3.org/ns/did/v1"],
        "authentication": [{
            "type": "Ed25519VerificationKey2020",
            "publicKeyMultibase": ed25519_public_multibase
        }],
        "keyAgreement": [{
            "type": "X25519KeyAgreementKey2020",
            "publicKeyMultibase": x25519_public_multibase
        }],
        "service": [{
            "type": "DIDCommMessaging",
            "serviceEndpoint": "http://localhost:8002/didcomm"
        }]
    });

    println!("=== did:peer:4 生成 ===\n");
    let (short_form, long_form) = create_did_peer_4(&input_doc);
    println!("Short Form DID:\n{}\n", short_form);
    println!("Long Form DID（先頭100文字）:\n{}...\n", &long_form[..100]);

    println!("=== 解決（Resolve）===\n");
    let resolved = resolve(&long_form);
    println!("Resolved DID Document:\n{}", serde_json::to_string_pretty(&resolved).unwrap());
}