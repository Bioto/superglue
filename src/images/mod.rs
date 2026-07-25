//! Image helpers: MIME detection, data URLs, lazy [`ImageRef`] resolution.

use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::openai::{ChatMessage, ContentPart, ImageUrl, MessageContent};

/// LLM-ready image payload cached by content hash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImagePayload {
    pub mime: String,
    pub data_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Session-scoped store mapping `sha256(file bytes)` → cached LLM payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageStore {
    entries: HashMap<String, ImagePayload>,
}

impl ImageStore {
    #[must_use]
    pub fn get(&self, hash: &str) -> Option<&ImagePayload> {
        self.entries.get(hash)
    }

    pub fn put(&mut self, hash: String, payload: ImagePayload) {
        self.entries.insert(hash, payload);
    }

    /// Hash bytes, cache payload on first insert, return the content hash.
    pub fn insert_from_bytes(&mut self, filename: &str, bytes: &[u8]) -> String {
        let hash = sha256_hex(bytes);
        if !self.entries.contains_key(&hash) {
            let mime = image_mime_from_filename(filename);
            let data_url = image_data_url(bytes, mime);
            self.put(
                hash.clone(),
                ImagePayload {
                    mime: mime.to_string(),
                    data_url,
                    filename: Some(filename.to_string()),
                },
            );
        }
        hash
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Find cached payload by exact data URL (for restoring lazy refs after API calls).
    #[must_use]
    pub fn lookup_by_data_url(&self, data_url: &str) -> Option<(String, &ImagePayload)> {
        self.entries
            .iter()
            .find(|(_, payload)| payload.data_url == data_url)
            .map(|(hash, payload)| (hash.clone(), payload))
    }
}

#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[must_use]
pub fn is_image_filename(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".gif")
        || lower.ends_with(".webp")
}

#[must_use]
pub fn image_mime_from_filename(filename: &str) -> &'static str {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else {
        "application/octet-stream"
    }
}

#[must_use]
pub fn image_data_url(bytes: &[u8], mime: &str) -> String {
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}

/// Build a user message with inline image bytes as `image_url` parts.
#[must_use]
pub fn message_with_image_bytes(
    text: Option<&str>,
    filename: &str,
    bytes: &[u8],
) -> ChatMessage {
    let mime = image_mime_from_filename(filename);
    let url = image_data_url(bytes, mime);
    let mut parts = Vec::new();
    if let Some(t) = text {
        parts.push(ContentPart::Text {
            text: t.to_string(),
        });
    }
    parts.push(ContentPart::ImageUrl {
        image_url: ImageUrl {
            url,
            detail: None,
        },
    });
    ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Parts(parts)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        refusal: None,
    }
}

/// Expand lazy image refs into provider-ready `image_url` parts.
#[must_use]
pub fn resolve_message_content(content: &MessageContent, store: &ImageStore) -> MessageContent {
    match content {
        MessageContent::Text(_) => content.clone(),
        MessageContent::Parts(parts) => {
            MessageContent::Parts(resolve_content_parts(parts, store))
        }
    }
}

/// Expand lazy image refs in a message slice for provider calls.
#[must_use]
pub fn resolve_messages(messages: Vec<ChatMessage>, store: &ImageStore) -> Vec<ChatMessage> {
    messages
        .into_iter()
        .map(|mut msg| {
            if let Some(content) = msg.content.take() {
                msg.content = Some(resolve_message_content(&content, store));
            }
            msg
        })
        .collect()
}

fn resolve_content_parts(parts: &[ContentPart], store: &ImageStore) -> Vec<ContentPart> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ContentPart::ImageRef { hash, filename } => {
                if let Some(payload) = store.get(hash) {
                    out.push(ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: payload.data_url.clone(),
                            detail: None,
                        },
                    });
                } else if let Some(name) = filename {
                    out.push(ContentPart::Text {
                        text: format!("[missing image: {name} ({hash})]"),
                    });
                } else {
                    out.push(ContentPart::Text {
                        text: format!("[missing image: {hash}]"),
                    });
                }
            }
            other => out.push(other.clone()),
        }
    }
    out
}

/// Text-only summary of multipart content (for compaction / transcripts).
#[must_use]
pub fn parts_text_for_summary(parts: &[ContentPart]) -> String {
    let mut chunks = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text } => {
                if !text.trim().is_empty() {
                    chunks.push(text.clone());
                }
            }
            ContentPart::ImageRef { filename, .. } => {
                if let Some(name) = filename {
                    chunks.push(format!("@{name}"));
                } else {
                    chunks.push("[image]".into());
                }
            }
            ContentPart::ImageUrl { .. } => chunks.push("[image]".into()),
            ContentPart::File { file } => {
                if let Some(name) = &file.filename {
                    chunks.push(format!("@{name}"));
                }
            }
            ContentPart::InputAudio { .. } => chunks.push("[audio]".into()),
        }
    }
    chunks.join("\n")
}

/// Replace resolved `image_url` parts with lazy refs when the URL matches the store.
#[must_use]
pub fn restore_image_refs_in_messages(
    messages: Vec<ChatMessage>,
    store: &ImageStore,
) -> Vec<ChatMessage> {
    messages
        .into_iter()
        .map(|mut msg| {
            if let Some(content) = msg.content.take() {
                msg.content = Some(restore_image_refs_in_content(&content, store));
            }
            msg
        })
        .collect()
}

fn restore_image_refs_in_content(content: &MessageContent, store: &ImageStore) -> MessageContent {
    match content {
        MessageContent::Text(_) => content.clone(),
        MessageContent::Parts(parts) => MessageContent::Parts(
            parts
                .iter()
                .map(|part| restore_image_ref_part(part, store))
                .collect(),
        ),
    }
}

fn restore_image_ref_part(part: &ContentPart, store: &ImageStore) -> ContentPart {
    match part {
        ContentPart::ImageUrl { image_url } => {
            if let Some((hash, payload)) = store.lookup_by_data_url(&image_url.url) {
                ContentPart::ImageRef {
                    hash,
                    filename: payload.filename.clone(),
                }
            } else {
                part.clone()
            }
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ImageDetail;

    #[test]
    fn restore_image_refs_after_resolve() {
        let bytes = b"round-trip";
        let mut store = ImageStore::default();
        let hash = store.insert_from_bytes("pic.png", bytes);
        let content = MessageContent::Parts(vec![ContentPart::ImageRef {
            hash: hash.clone(),
            filename: Some("pic.png".into()),
        }]);
        let resolved = resolve_message_content(&content, &store);
        let MessageContent::Parts(resolved_parts) = resolved else {
            panic!("expected parts");
        };
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(resolved_parts)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            refusal: None,
        };
        let restored = restore_image_refs_in_messages(vec![msg], &store);
        let Some(MessageContent::Parts(parts)) = &restored[0].content else {
            panic!("expected parts");
        };
        assert!(matches!(parts[0], ContentPart::ImageRef { .. }));
    }

    #[test]
    fn insert_from_bytes_caches_by_hash() {
        let bytes = b"\x89PNG\r\n\x1a\n";
        let mut store = ImageStore::default();
        let h1 = store.insert_from_bytes("a.png", bytes);
        let h2 = store.insert_from_bytes("b.png", bytes);
        assert_eq!(h1, h2);
        assert!(store.get(&h1).is_some());
        assert!(store
            .get(&h1)
            .unwrap()
            .data_url
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn resolve_image_ref_to_image_url() {
        let bytes = b"fake-png";
        let mut store = ImageStore::default();
        let hash = store.insert_from_bytes("shot.png", bytes);
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "what is this?".into(),
            },
            ContentPart::ImageRef {
                hash,
                filename: Some("shot.png".into()),
            },
        ]);
        let resolved = resolve_message_content(&content, &store);
        let MessageContent::Parts(parts) = resolved else {
            panic!("expected parts");
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[1], ContentPart::ImageUrl { .. }));
    }

    #[test]
    fn message_with_image_bytes_serialises_image_url() {
        let msg = message_with_image_bytes(Some("look"), "x.png", b"png");
        let v = serde_json::to_value(&msg).unwrap();
        let parts = v["content"].as_array().unwrap();
        assert_eq!(parts[1]["type"], "image_url");
        assert!(parts[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
        let _ = ImageDetail::Auto; // keep import used in openai tests parity
    }
}
