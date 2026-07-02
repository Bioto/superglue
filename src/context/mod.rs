//! Context optimization — condensing config and AAAK pure encoding.

mod aaak;
mod condense;
mod summarize;

pub(crate) use aaak::{
    message_text, passthrough_aaak_context, passthrough_at_messages, transcript_from_messages,
    COMPRESS_SYSTEM, COMPRESS_USER_PREFIX,
};
pub use aaak::AaakCompressor;
pub use condense::condense_tool_round;
pub use summarize::SummarizeContextConfig;
