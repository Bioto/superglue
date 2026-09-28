//! Budget enforcement helpers.

use crate::gateway::db::Database;
use crate::gateway::error::{GatewayError, GatewayResult};

/// Lazy-reset the user's budget period and reject if enforce mode and over limit.
pub fn check_budget(db: &Database, user_id: &str) -> GatewayResult<()> {
    let (enforce, max_budget, spend) = db.prepare_user_budget(user_id)?;
    if enforce && spend >= max_budget {
        return Err(GatewayError::budget_exceeded(format!(
            "user {user_id} has exceeded budget (${spend:.4} >= ${max_budget:.4})"
        )));
    }
    Ok(())
}
