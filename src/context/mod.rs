//! Context optimization — condensing config and AAAK pure encoding.

mod aaak;
mod condense;
mod summarize;

pub use aaak::AaakCompressor;
pub(crate) use aaak::{
    COMPRESS_SYSTEM, COMPRESS_USER_PREFIX, message_text, passthrough_aaak_context,
    passthrough_at_messages, transcript_from_messages,
};
pub use condense::condense_tool_round;
pub use summarize::SummarizeContextConfig;
