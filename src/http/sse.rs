//! Minimal Server-Sent Events (SSE) framing parser for LLM-style streams.

use std::time::{Duration, Instant};

use crate::http::error::Error;

/// Wait this long for `[DONE]` after a terminal chunk when the socket stays open.
const STREAM_TAIL_GRACE: Duration = Duration::from_secs(1);

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

    /// True when the unfinished buffer is blank or only SSE comment lines.
    fn partial_is_heartbeat(&self) -> bool {
        let buf = self.buf.trim_start_matches(['\r', '\n']);
        buf.split('\n').all(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            line.is_empty() || line.starts_with(':')
        })
    }
}

impl SseEvent {
    fn is_empty_heartbeat(&self) -> bool {
        self.event.is_none() && self.id.is_none() && self.data.trim().is_empty()
    }
}

/// What to do with one SSE chunk after heartbeats are separated from model bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatAction {
    /// The chunk has model data. Parse `events`.
    Model,
    /// Provider heartbeat. Keep reading.
    Ignore,
    /// A terminal chunk was already seen, and only heartbeats followed.
    Finish,
    /// No model bytes arrived within the idle window.
    Stall,
}

/// Idle clock that provider comment keepalives must not reset.
///
/// OpenRouter sends `: OPENROUTER PROCESSING` while a model is queued or silent.
/// Those bytes keep the socket idle timer alive. This clock moves only on model data.
#[derive(Debug)]
pub struct HeartbeatWatch {
    idle: Duration,
    tail: Duration,
    last_progress: Instant,
}

impl HeartbeatWatch {
    /// `idle` is the configured stream idle timeout.
    #[must_use]
    pub fn new(idle: Duration) -> Self {
        Self::with_tail(idle, STREAM_TAIL_GRACE)
    }

    #[must_use]
    pub fn with_tail(idle: Duration, tail: Duration) -> Self {
        Self {
            idle,
            tail,
            last_progress: Instant::now(),
        }
    }

    /// Classify `events` from the latest chunk.
    ///
    /// `saw_terminal` is true when an earlier chunk already carried a finish signal.
    pub fn observe(
        &mut self,
        parser: &SseParser,
        events: &[SseEvent],
        saw_terminal: bool,
    ) -> HeartbeatAction {
        if !is_heartbeat_chunk(parser, events) {
            self.last_progress = Instant::now();
            return HeartbeatAction::Model;
        }
        let limit = if saw_terminal { self.tail } else { self.idle };
        if self.last_progress.elapsed() >= limit {
            if saw_terminal {
                HeartbeatAction::Finish
            } else {
                HeartbeatAction::Stall
            }
        } else {
            HeartbeatAction::Ignore
        }
    }
}

/// True when this chunk is only SSE comments, blank events, or an unfinished comment.
pub fn is_heartbeat_chunk(parser: &SseParser, events: &[SseEvent]) -> bool {
    events.iter().all(SseEvent::is_empty_heartbeat) && parser.partial_is_heartbeat()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

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

    #[tokio::test]
    async fn a_split_comment_does_not_reset_idle() {
        let mut parser = SseParser::new();
        let mut watch =
            HeartbeatWatch::with_tail(Duration::from_millis(50), Duration::from_secs(5));
        tokio::time::sleep(Duration::from_millis(40)).await;
        let events = parser.push_str(": OPEN").unwrap();
        assert_eq!(
            watch.observe(&parser, &events, false),
            HeartbeatAction::Ignore
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
        let events = parser.push_str("ROUTER PROCESSING\n\n").unwrap();
        assert_eq!(
            watch.observe(&parser, &events, false),
            HeartbeatAction::Stall
        );
    }

    #[tokio::test]
    async fn model_bytes_reset_the_idle_window() {
        let mut parser = SseParser::new();
        let mut watch =
            HeartbeatWatch::with_tail(Duration::from_millis(50), Duration::from_secs(5));
        let partial = parser.push_str("data: {\"a\"").unwrap();
        assert_eq!(
            watch.observe(&parser, &partial, false),
            HeartbeatAction::Model
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
        let events = parser.push_str("\":1}\n\n").unwrap();
        assert_eq!(
            watch.observe(&parser, &events, false),
            HeartbeatAction::Model
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
        let events = parser.push_str(": OPENROUTER PROCESSING\n\n").unwrap();
        assert_eq!(
            watch.observe(&parser, &events, false),
            HeartbeatAction::Ignore
        );
    }

    #[tokio::test]
    async fn comments_after_a_terminal_chunk_finish_the_stream() {
        let mut parser = SseParser::new();
        let mut watch =
            HeartbeatWatch::with_tail(Duration::from_secs(5), Duration::from_millis(30));
        let events = parser.push_str("data: {\"choices\":[]}\n\n").unwrap();
        assert_eq!(
            watch.observe(&parser, &events, false),
            HeartbeatAction::Model
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
        let events = parser.push_str(": OPENROUTER PROCESSING\n\n").unwrap();
        assert_eq!(
            watch.observe(&parser, &events, true),
            HeartbeatAction::Finish
        );
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
