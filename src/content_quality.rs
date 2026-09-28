//! Shared heuristics for bot-block and low-signal web text.

use crate::http::HttpClient;
use crate::providers::{ProviderCredentials, ProviderId};
use crate::systemone::{Noul, Questions, SystemOneRequest, clip_state, noul_yes, system_one};

/// Minimum alphanumeric characters for a page body to count as substantive content.
pub const MIN_ALPHANUMERIC_CHARS: usize = 200;

/// Shared block / challenge phrases.
pub const BLOCK_PHRASES: &[&str] = &[
    "captcha",
    "cf-browser-verification",
    "just a moment",
    "unusual traffic",
    "access denied",
    "enable javascript",
    "are you a robot",
    "checking your browser",
    "attention required",
    "request blocked",
];

const REGION_PICKER_COUNTRIES: &[&str] = &[
    "argentina",
    "australia",
    "austria",
    "belgium",
    "brazil",
    "bulgaria",
    "canada",
    "chile",
    "china",
    "colombia",
    "denmark",
    "france",
    "germany",
    "india",
    "ireland",
    "italy",
    "japan",
    "mexico",
    "netherlands",
    "poland",
    "spain",
    "sweden",
    "switzerland",
    "united kingdom",
    "united states",
];

const GRAY_MAX_ALNUM: usize = 400;
const PAGE_JEV_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(400);
const PAGE_NOUL: f64 = 0.5;

/// Returns true when `text` looks like a bot-block page, region picker, or other non-content chrome.
#[must_use]
pub fn is_low_signal_web_text(text: &str) -> bool {
    match classify_web_text(text) {
        WebTextClass::Empty | WebTextClass::Phrase | WebTextClass::Region | WebTextClass::Thin => {
            true
        }
        WebTextClass::Ok | WebTextClass::Gray => false,
    }
}

/// True when a shared block phrase is present.
#[must_use]
pub fn has_block_phrase(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    BLOCK_PHRASES.iter().any(|phrase| lower.contains(phrase))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WebTextClass {
    Empty,
    Phrase,
    Region,
    Thin,
    Gray,
    Ok,
}

fn classify_web_text(text: &str) -> WebTextClass {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return WebTextClass::Empty;
    }
    let lower = trimmed.to_ascii_lowercase();
    if BLOCK_PHRASES.iter().any(|phrase| lower.contains(phrase)) {
        return WebTextClass::Phrase;
    }
    if lower.contains("all regions") {
        let head: String = lower.chars().take(2000).collect();
        let country_hits = REGION_PICKER_COUNTRIES
            .iter()
            .filter(|country| head.contains(*country))
            .count();
        if country_hits >= 3 {
            return WebTextClass::Region;
        }
    }
    let alnum = trimmed.chars().filter(|c| c.is_alphanumeric()).count();
    if alnum < MIN_ALPHANUMERIC_CHARS {
        return WebTextClass::Thin;
    }
    if alnum < GRAY_MAX_ALNUM {
        return WebTextClass::Gray;
    }
    WebTextClass::Ok
}

/// Gray-zone Noul for thin pages without a phrase hit. Timeout keeps the deterministic result.
pub async fn is_low_signal_web_text_judged(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    text: &str,
) -> bool {
    let class = classify_web_text(text);
    match class {
        WebTextClass::Empty | WebTextClass::Phrase | WebTextClass::Region | WebTextClass::Thin => {
            true
        }
        WebTextClass::Ok => false,
        WebTextClass::Gray => {
            if !credentials.has_key(ProviderId::TypeSafe) {
                return false;
            }
            match judge_page_usable(http, credentials, text).await {
                Some(usable) => !usable,
                None => false,
            }
        }
    }
}

async fn judge_page_usable(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    text: &str,
) -> Option<bool> {
    let mut questions = Questions::new();
    questions.insert(
        "usable",
        Noul::new("Is this usable page content, not a block, captcha, or picker?").criteria(
            "Yes, real article or documentation text.",
            "No, chrome or a block page.",
        ),
    );
    let request =
        SystemOneRequest::new(clip_state(text, 400), questions).with_model("typesafe:jev-latest");
    let work = system_one(http, credentials, request);
    let response = tokio::time::timeout(PAGE_JEV_TIMEOUT, work)
        .await
        .ok()?
        .ok()?;
    let answer = response.answer("usable")?;
    Some(noul_yes(answer, PAGE_NOUL))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_low_signal() {
        assert!(is_low_signal_web_text(""));
        assert!(is_low_signal_web_text("   \n  "));
    }

    #[test]
    fn captcha_is_low_signal() {
        assert!(is_low_signal_web_text(
            "Please complete the captcha to continue browsing this site."
        ));
    }

    #[test]
    fn article_is_not_low_signal() {
        let article = "A ".repeat(200);
        assert!(!is_low_signal_web_text(&article));
    }

    fn gray_page() -> String {
        "word ".repeat(70)
    }

    #[test]
    fn gray_page_is_not_deterministic_low_signal() {
        let text = gray_page();
        assert_eq!(classify_web_text(&text), WebTextClass::Gray);
        assert!(!is_low_signal_web_text(&text));
    }

    #[tokio::test]
    async fn gray_page_jev_no_marks_low_signal() {
        use crate::http::{ClientConfig, RetryPolicy};
        use crate::providers::ProviderId;
        use serde_json::json;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "jev-latest",
                "answers": { "usable": { "type": "noul", "noul": 0.1 } },
                "usage": { "input_tokens": 4, "output_tokens": 1 }
            })))
            .mount(&server)
            .await;

        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::TypeSafe, "ts-test");
        creds.insert_base_url(ProviderId::TypeSafe, server.uri());
        let http = HttpClient::new(ClientConfig {
            retry: RetryPolicy {
                max_retries: 0,
                ..RetryPolicy::default()
            },
            ..ClientConfig::default()
        })
        .expect("http");
        assert!(is_low_signal_web_text_judged(&http, &creds, &gray_page()).await);
    }

    #[tokio::test]
    async fn gray_page_timeout_keeps_deterministic() {
        use crate::http::{ClientConfig, RetryPolicy};
        use crate::providers::ProviderId;
        use serde_json::json;
        use std::time::Duration;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({
                        "model": "jev-latest",
                        "answers": { "usable": { "type": "noul", "noul": 0.1 } },
                        "usage": { "input_tokens": 4, "output_tokens": 1 }
                    }))
                    .set_delay(Duration::from_secs(2)),
            )
            .mount(&server)
            .await;

        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::TypeSafe, "ts-test");
        creds.insert_base_url(ProviderId::TypeSafe, server.uri());
        let http = HttpClient::new(ClientConfig {
            retry: RetryPolicy {
                max_retries: 0,
                ..RetryPolicy::default()
            },
            ..ClientConfig::default()
        })
        .expect("http");
        assert!(!is_low_signal_web_text_judged(&http, &creds, &gray_page()).await);
    }
}
