//! Minimal Server-Sent Events (SSE) framing parser for LLM-style streams.

use crate::http::error::Error;

/// One SSE event after a blank line delimiter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub id: Option<String>,
    /// Concatenated `data:` lines with `\n` between them (SSE spec).
    pub data: String,
}

/// Incremental SSE parser: feed UTF-8 text; emitted events are removed from the internal buffer.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: String,
}

impl SseParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push more text from the wire; returns completed events.
    pub fn push_str(&mut self, chunk: &str) -> Result<Vec<SseEvent>, Error> {
        self.buf.push_str(chunk);
        self.drain_complete_events()
    }

    fn drain_complete_events(&mut self) -> Result<Vec<SseEvent>, Error> {
        let mut out = Vec::new();
        let mut consumed = 0;
        while let Some(relative_pos) = self.buf[consumed..].find("\n\n") {
            let pos = consumed + relative_pos;
            if let Some(ev) = Self::parse_event_block(&self.buf[consumed..pos])? {
                out.push(ev);
            }
            consumed = pos + 2;
        }
        if consumed > 0 {
            self.buf = self.buf.split_off(consumed);
        }
        Ok(out)
    }

    fn parse_event_block(block: &str) -> Result<Option<SseEvent>, Error> {
        if block.is_empty() {
            return Ok(None);
        }
        let mut event = None;
        let mut id = None;
        let mut data = String::new();
        let mut has_data = false;
        for line in block.split('\n') {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() {
                continue;
            }
            if line.starts_with(':') {
                // Comment / heartbeat — ignore.
                continue;
            }
            if let Some(v) = line.strip_prefix("event:") {
                event = Some(v.trim().to_string());
            } else if let Some(v) = line.strip_prefix("id:") {
                id = Some(v.trim().to_string());
            } else if let Some(v) = line.strip_prefix("data:") {
                if has_data {
                    data.push('\n');
                }
                data.push_str(v.trim_start());
                has_data = true;
            } else {
                return Err(Error::SseParse(format!("unrecognized SSE line: {line}")));
            }
        }
        Ok(Some(SseEvent { event, id, data }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_data_event() {
        let mut p = SseParser::new();
        let evs = p.push_str("data: hello\n\n").unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "hello");
    }

    #[test]
    fn parses_event_and_multiline_data() {
        let mut p = SseParser::new();
        let chunk = "event: delta\nid: 1\ndata: line1\ndata: line2\n\n";
        let evs = p.push_str(chunk).unwrap();
        assert_eq!(evs[0].event.as_deref(), Some("delta"));
        assert_eq!(evs[0].id.as_deref(), Some("1"));
        assert_eq!(evs[0].data, "line1\nline2");
    }

    #[test]
    fn split_across_chunks() {
        let mut p = SseParser::new();
        assert!(p.push_str("data: a").unwrap().is_empty());
        let evs = p.push_str("\n\n").unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "a");
    }

    #[test]
    fn parses_multiple_events_from_one_chunk() {
        let mut p = SseParser::new();
        let evs = p.push_str("data: one\n\ndata: two\n\ndata: three").unwrap();
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "one");
        assert_eq!(evs[1].data, "two");
        let final_event = p.push_str("\n\n").unwrap();
        assert_eq!(final_event.len(), 1);
        assert_eq!(final_event[0].data, "three");
    }

    #[test]
    fn parses_error_event_type() {
        let mut p = SseParser::new();
        let evs = p
            .push_str("event: error\ndata: {\"error\":\"boom\"}\n\n")
            .unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event.as_deref(), Some("error"));
        assert_eq!(evs[0].data, "{\"error\":\"boom\"}");
    }
}
