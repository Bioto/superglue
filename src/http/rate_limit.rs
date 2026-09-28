use std::num::NonZeroU32;
use std::sync::Arc;

use governor::middleware::NoOpMiddleware;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter, clock::DefaultClock};

/// Direct (non-keyed) rate limiter used before each HTTP request.
pub type DirectRateLimiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

/// Build a simple per-second quota limiter.
#[must_use]
pub fn direct_per_second(qps: NonZeroU32) -> Arc<DirectRateLimiter> {
    Arc::new(RateLimiter::direct(Quota::per_second(qps)))
}
