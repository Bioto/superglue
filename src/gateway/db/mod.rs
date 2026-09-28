//! SQLite persistence for gateway keys, users, budgets, and usage.

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::gateway::auth::hash_key;
use crate::gateway::error::{GatewayError, GatewayResult};

const SCHEMA_VERSION: i32 = 2;

const MIGRATION_V1: &str = "
CREATE TABLE IF NOT EXISTS budgets (
    id TEXT PRIMARY KEY NOT NULL,
    max_budget REAL NOT NULL,
    duration_sec INTEGER NOT NULL,
    enforce INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY NOT NULL,
    alias TEXT,
    budget_id TEXT REFERENCES budgets(id),
    spend REAL NOT NULL DEFAULT 0.0,
    next_budget_reset_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY NOT NULL,
    key_hash BLOB NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    name TEXT,
    user_id TEXT NOT NULL REFERENCES users(id),
    active INTEGER NOT NULL DEFAULT 1,
    expires_at TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_key_models (
    key_id TEXT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    model_pattern TEXT NOT NULL,
    PRIMARY KEY (key_id, model_pattern)
);

CREATE TABLE IF NOT EXISTS usage_logs (
    id TEXT PRIMARY KEY NOT NULL,
    key_id TEXT,
    user_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    request_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_reset_logs (
    id TEXT PRIMARY KEY NOT NULL,
    user_id TEXT NOT NULL,
    budget_id TEXT NOT NULL,
    previous_spend REAL NOT NULL,
    reset_at TEXT NOT NULL
);
";

const MIGRATION_V2: &str = "
CREATE INDEX IF NOT EXISTS idx_usage_logs_created_at ON usage_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_logs_user_created ON usage_logs(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_logs_key_created ON usage_logs(key_id, created_at);
";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetResetLogRecord {
    pub id: String,
    pub user_id: String,
    pub budget_id: String,
    pub previous_spend: f64,
    pub reset_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummaryTotals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummaryRow {
    pub key: String,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummary {
    pub totals: UsageSummaryTotals,
    pub groups: Vec<UsageSummaryRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageSummaryGroupBy {
    User,
    Model,
    Key,
    Day,
}

impl UsageSummaryGroupBy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "user" => Some(Self::User),
            "model" => Some(Self::Model),
            "key" => Some(Self::Key),
            "day" => Some(Self::Day),
            _ => None,
        }
    }

    fn group_sql(&self) -> &'static str {
        match self {
            Self::User => "user_id",
            Self::Model => "model",
            Self::Key => "COALESCE(key_id, '')",
            Self::Day => "substr(created_at, 1, 10)",
        }
    }
}

/// Thread-safe SQLite handle (single connection + WAL).
#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRecord {
    pub id: String,
    pub alias: Option<String>,
    pub budget_id: Option<String>,
    pub spend: f64,
    pub next_budget_reset_at: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetRecord {
    pub id: String,
    pub max_budget: f64,
    pub duration_sec: i64,
    pub enforce: bool,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct ApiKeyRecord {
    pub id: String,
    pub key_prefix: String,
    pub name: Option<String>,
    pub user_id: String,
    pub active: i32,
    pub expires_at: Option<String>,
    pub metadata_json: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyListItem {
    pub id: String,
    pub key_prefix: String,
    pub name: Option<String>,
    pub user_id: String,
    pub active: bool,
    pub expires_at: Option<String>,
    pub metadata_json: Option<String>,
    pub allowed_models: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub id: String,
    pub key_id: Option<String>,
    pub user_id: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cost_usd: f64,
    pub request_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct CreateKeyResult {
    pub id: String,
    pub plaintext_key: String,
    pub key_prefix: String,
}

impl Database {
    /// Open (or create) the database at `path` and run migrations.
    pub fn open(path: impl AsRef<Path>) -> GatewayResult<Self> {
        let conn = Connection::open(path)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        let _: String = conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
        conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA temp_store = MEMORY;")?;
        run_migrations(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn lock(&self) -> GatewayResult<MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| GatewayError::Internal("database lock poisoned".into()))
    }

    /// Run a synchronous database operation on the blocking thread pool.
    pub async fn run_blocking<F, T>(&self, f: F) -> GatewayResult<T>
    where
        F: FnOnce(&Self) -> GatewayResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let db = self.clone();
        tokio::task::spawn_blocking(move || f(&db))
            .await
            .map_err(|_| GatewayError::Internal("database task join failed".into()))?
    }

    /// Lightweight health check.
    pub fn ping(&self) -> GatewayResult<()> {
        self.lock()?.query_row("SELECT 1", [], |_| Ok(()))?;
        Ok(())
    }

    pub fn create_budget(
        &self,
        max_budget: f64,
        duration_sec: i64,
        enforce: bool,
    ) -> GatewayResult<BudgetRecord> {
        let conn = self.lock()?;
        let id = Uuid::new_v4().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO budgets (id, max_budget, duration_sec, enforce, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, max_budget, duration_sec, i32::from(enforce), created_at],
        )?;
        Ok(BudgetRecord {
            id,
            max_budget,
            duration_sec,
            enforce,
            created_at,
        })
    }

    pub fn list_budgets(&self) -> GatewayResult<Vec<BudgetRecord>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, max_budget, duration_sec, enforce, created_at FROM budgets ORDER BY created_at",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(BudgetRecord {
                id: row.get(0)?,
                max_budget: row.get(1)?,
                duration_sec: row.get(2)?,
                enforce: row.get::<_, i32>(3)? != 0,
                created_at: row.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_budget(
        &self,
        budget_id: &str,
        max_budget: Option<f64>,
        duration_sec: Option<i64>,
        enforce: Option<bool>,
    ) -> GatewayResult<BudgetRecord> {
        let conn = self.lock()?;
        if let Some(v) = max_budget {
            let n = conn.execute(
                "UPDATE budgets SET max_budget = ?1 WHERE id = ?2",
                params![v, budget_id],
            )?;
            if n == 0 {
                return Err(GatewayError::not_found(format!(
                    "budget {budget_id} not found"
                )));
            }
        }
        if let Some(v) = duration_sec {
            let n = conn.execute(
                "UPDATE budgets SET duration_sec = ?1 WHERE id = ?2",
                params![v, budget_id],
            )?;
            if n == 0 {
                return Err(GatewayError::not_found(format!(
                    "budget {budget_id} not found"
                )));
            }
        }
        if let Some(v) = enforce {
            let n = conn.execute(
                "UPDATE budgets SET enforce = ?1 WHERE id = ?2",
                params![i32::from(v), budget_id],
            )?;
            if n == 0 {
                return Err(GatewayError::not_found(format!(
                    "budget {budget_id} not found"
                )));
            }
        }
        conn.query_row(
            "SELECT id, max_budget, duration_sec, enforce, created_at FROM budgets WHERE id = ?1",
            params![budget_id],
            map_budget,
        )
        .optional()?
        .ok_or_else(|| GatewayError::not_found(format!("budget {budget_id} not found")))
    }

    /// Delete a budget tier and clear `budget_id` on users that referenced it.
    pub fn delete_budget(&self, budget_id: &str) -> GatewayResult<u32> {
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        let users_cleared = tx.execute(
            "UPDATE users SET budget_id = NULL, next_budget_reset_at = NULL WHERE budget_id = ?1",
            params![budget_id],
        )? as u32;
        let deleted = tx.execute("DELETE FROM budgets WHERE id = ?1", params![budget_id])?;
        if deleted == 0 {
            return Err(GatewayError::not_found(format!(
                "budget {budget_id} not found"
            )));
        }
        tx.commit()?;
        Ok(users_cleared)
    }

    pub fn create_user(
        &self,
        user_id: &str,
        alias: Option<&str>,
        budget_id: Option<&str>,
    ) -> GatewayResult<UserRecord> {
        let conn = self.lock()?;
        let created_at = chrono::Utc::now().to_rfc3339();
        let next_reset = budget_id
            .map(|bid| compute_next_reset(&conn, bid))
            .transpose()?;
        conn.execute(
            "INSERT INTO users (id, alias, budget_id, spend, next_budget_reset_at, created_at) VALUES (?1, ?2, ?3, 0.0, ?4, ?5)",
            params![user_id, alias, budget_id, next_reset, created_at],
        )?;
        Ok(UserRecord {
            id: user_id.to_string(),
            alias: alias.map(str::to_string),
            budget_id: budget_id.map(str::to_string),
            spend: 0.0,
            next_budget_reset_at: next_reset,
            created_at,
        })
    }

    pub fn get_user(&self, user_id: &str) -> GatewayResult<Option<UserRecord>> {
        let conn = self.lock()?;
        conn.query_row(
            "SELECT id, alias, budget_id, spend, next_budget_reset_at, created_at FROM users WHERE id = ?1",
            params![user_id],
            map_user,
        )
        .optional()
        .map_err(Into::into)
    }

    pub fn list_users(&self) -> GatewayResult<Vec<UserRecord>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, alias, budget_id, spend, next_budget_reset_at, created_at FROM users ORDER BY created_at",
        )?;
        let rows = stmt.query_map([], map_user)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_user(
        &self,
        user_id: &str,
        alias: Option<Option<&str>>,
        budget_id: Option<Option<&str>>,
    ) -> GatewayResult<UserRecord> {
        let conn = self.lock()?;
        if let Some(a) = alias {
            conn.execute(
                "UPDATE users SET alias = ?1 WHERE id = ?2",
                params![a, user_id],
            )?;
        }
        if let Some(b) = budget_id {
            let next_reset = b.map(|bid| compute_next_reset(&conn, bid)).transpose()?;
            conn.execute(
                "UPDATE users SET budget_id = ?1, next_budget_reset_at = ?2 WHERE id = ?3",
                params![b, next_reset, user_id],
            )?;
        }
        conn.query_row(
            "SELECT id, alias, budget_id, spend, next_budget_reset_at, created_at FROM users WHERE id = ?1",
            params![user_id],
            map_user,
        )
        .optional()?
        .ok_or_else(|| GatewayError::not_found(format!("user {user_id} not found")))
    }

    /// Delete a user and all of their API keys.
    pub fn delete_user(&self, user_id: &str) -> GatewayResult<u32> {
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        let keys_deleted =
            tx.execute("DELETE FROM api_keys WHERE user_id = ?1", params![user_id])? as u32;
        let users_deleted = tx.execute("DELETE FROM users WHERE id = ?1", params![user_id])?;
        if users_deleted == 0 {
            return Err(GatewayError::not_found(format!("user {user_id} not found")));
        }
        tx.commit()?;
        Ok(keys_deleted)
    }

    pub fn user_exists(&self, user_id: &str) -> GatewayResult<bool> {
        let conn = self.lock()?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM users WHERE id = ?1",
            params![user_id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    pub fn create_api_key(
        &self,
        name: Option<&str>,
        user_id: &str,
        allowed_models: &[String],
        expires_at: Option<&str>,
        metadata_json: Option<&str>,
    ) -> GatewayResult<CreateKeyResult> {
        if allowed_models.is_empty() {
            return Err(GatewayError::bad_request(
                "allowed_models must contain at least one pattern",
            ));
        }
        if !self.user_exists(user_id)? {
            return Err(GatewayError::bad_request(format!(
                "user {user_id} does not exist"
            )));
        }

        let plaintext = format!("sgw-{}", Uuid::new_v4().simple());
        let key_hash = hash_key(&plaintext);
        let key_prefix = plaintext.chars().take(12).collect::<String>() + "...";
        let id = Uuid::new_v4().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();

        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO api_keys (id, key_hash, key_prefix, name, user_id, active, expires_at, metadata_json, created_at) VALUES (?1, ?2, ?3, ?4, ?5, 1, ?6, ?7, ?8)",
            params![id, key_hash.as_slice(), key_prefix, name, user_id, expires_at, metadata_json, created_at],
        )?;
        for pattern in allowed_models {
            tx.execute(
                "INSERT INTO api_key_models (key_id, model_pattern) VALUES (?1, ?2)",
                params![id, pattern],
            )?;
        }
        tx.commit()?;

        Ok(CreateKeyResult {
            id,
            plaintext_key: plaintext,
            key_prefix,
        })
    }

    pub fn list_api_keys(&self) -> GatewayResult<Vec<ApiKeyListItem>> {
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT id, key_prefix, name, user_id, active, expires_at, metadata_json, created_at FROM api_keys ORDER BY created_at",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i32>(4)? != 0,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, key_prefix, name, user_id, active, expires_at, metadata_json, created_at) =
                row?;
            let models = self.list_key_models_in_conn(&conn, &id)?;
            out.push(ApiKeyListItem {
                id,
                key_prefix,
                name,
                user_id,
                active,
                expires_at,
                metadata_json,
                allowed_models: models,
                created_at,
            });
        }
        Ok(out)
    }

    pub fn lookup_api_key_by_hash(&self, hash: &[u8; 32]) -> GatewayResult<Option<ApiKeyRecord>> {
        let conn = self.lock()?;
        conn.query_row(
            "SELECT id, key_prefix, name, user_id, active, expires_at, metadata_json, created_at FROM api_keys WHERE key_hash = ?1",
            params![hash.as_slice()],
            map_api_key,
        )
        .optional()
        .map_err(Into::into)
    }

    /// Look up a virtual key and its model allowlist while holding one connection lock.
    pub fn lookup_api_key_with_models(
        &self,
        hash: &[u8; 32],
    ) -> GatewayResult<Option<(ApiKeyRecord, Vec<String>)>> {
        let conn = self.lock()?;
        let Some(record) = conn
            .query_row(
                "SELECT id, key_prefix, name, user_id, active, expires_at, metadata_json, created_at FROM api_keys WHERE key_hash = ?1",
                params![hash.as_slice()],
                map_api_key,
            )
            .optional()?
        else {
            return Ok(None);
        };
        let models = self.list_key_models_in_conn(&conn, &record.id)?;
        Ok(Some((record, models)))
    }

    pub fn list_key_models(&self, key_id: &str) -> GatewayResult<Vec<String>> {
        let conn = self.lock()?;
        self.list_key_models_in_conn(&conn, key_id)
    }

    fn list_key_models_in_conn(
        &self,
        conn: &Connection,
        key_id: &str,
    ) -> GatewayResult<Vec<String>> {
        let mut stmt = conn.prepare(
            "SELECT model_pattern FROM api_key_models WHERE key_id = ?1 ORDER BY model_pattern",
        )?;
        let rows = stmt.query_map(params![key_id], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_api_key(
        &self,
        key_id: &str,
        active: Option<bool>,
        allowed_models: Option<&[String]>,
        expires_at: Option<Option<&str>>,
        metadata: Option<Option<serde_json::Value>>,
    ) -> GatewayResult<ApiKeyListItem> {
        let conn = self.lock()?;
        let exists = conn
            .query_row(
                "SELECT 1 FROM api_keys WHERE id = ?1",
                params![key_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !exists {
            return Err(GatewayError::not_found(format!("key {key_id} not found")));
        }
        let tx = conn.unchecked_transaction()?;
        if let Some(a) = active {
            tx.execute(
                "UPDATE api_keys SET active = ?1 WHERE id = ?2",
                params![i32::from(a), key_id],
            )?;
        }
        if let Some(exp) = expires_at {
            tx.execute(
                "UPDATE api_keys SET expires_at = ?1 WHERE id = ?2",
                params![exp, key_id],
            )?;
        }
        if let Some(meta) = metadata {
            let json = match meta {
                None => None,
                Some(value) => {
                    let existing: Option<String> = tx
                        .query_row(
                            "SELECT metadata_json FROM api_keys WHERE id = ?1",
                            params![key_id],
                            |row| row.get(0),
                        )
                        .optional()?;
                    let merged = merge_metadata_json(existing.as_deref(), &value);
                    Some(
                        serde_json::to_string(&merged)
                            .map_err(|e| GatewayError::bad_request(e.to_string()))?,
                    )
                }
            };
            tx.execute(
                "UPDATE api_keys SET metadata_json = ?1 WHERE id = ?2",
                params![json, key_id],
            )?;
        }
        if let Some(models) = allowed_models {
            if models.is_empty() {
                return Err(GatewayError::bad_request(
                    "allowed_models must contain at least one pattern",
                ));
            }
            tx.execute(
                "DELETE FROM api_key_models WHERE key_id = ?1",
                params![key_id],
            )?;
            for pattern in models {
                tx.execute(
                    "INSERT INTO api_key_models (key_id, model_pattern) VALUES (?1, ?2)",
                    params![key_id, pattern],
                )?;
            }
        }
        tx.commit()?;
        drop(conn);
        self.list_api_keys()?
            .into_iter()
            .find(|k| k.id == key_id)
            .ok_or_else(|| GatewayError::not_found(format!("key {key_id} not found")))
    }

    pub fn delete_api_key(&self, key_id: &str) -> GatewayResult<()> {
        let conn = self.lock()?;
        let n = conn.execute("DELETE FROM api_keys WHERE id = ?1", params![key_id])?;
        if n == 0 {
            return Err(GatewayError::not_found(format!("key {key_id} not found")));
        }
        Ok(())
    }

    pub fn get_budget(&self, budget_id: &str) -> GatewayResult<Option<BudgetRecord>> {
        let conn = self.lock()?;
        conn.query_row(
            "SELECT id, max_budget, duration_sec, enforce, created_at FROM budgets WHERE id = ?1",
            params![budget_id],
            map_budget,
        )
        .optional()
        .map_err(Into::into)
    }

    /// Lazy budget reset + pre-request enforce check. Returns `(enforce, max_budget, current_spend)`.
    ///
    /// The common path is a `SELECT` only. A write transaction runs when the
    /// reset window is missing (first setup) or already past.
    pub fn prepare_user_budget(&self, user_id: &str) -> GatewayResult<(bool, f64, f64)> {
        let conn = self.lock()?;
        let (budget_id, spend, next_reset) = conn
            .query_row(
                "SELECT budget_id, spend, next_budget_reset_at FROM users WHERE id = ?1",
                params![user_id],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .map_err(|_| GatewayError::not_found(format!("user {user_id} not found")))?;

        let Some(budget_id) = budget_id else {
            return Ok((false, f64::MAX, spend));
        };

        let (max_budget, duration_sec, enforce) = conn
            .query_row(
                "SELECT max_budget, duration_sec, enforce FROM budgets WHERE id = ?1",
                params![budget_id],
                |row| {
                    Ok((
                        row.get::<_, f64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i32>(2)? != 0,
                    ))
                },
            )
            .map_err(|_| GatewayError::not_found(format!("budget {budget_id} not found")))?;

        let now = chrono::Utc::now();
        let now_rfc = now.to_rfc3339();
        let reset_due = next_reset
            .as_deref()
            .is_none_or(|reset_at| reset_at <= now_rfc.as_str());
        if !reset_due {
            return Ok((enforce, max_budget, spend));
        }

        let tx = conn.unchecked_transaction()?;
        let mut current_spend = spend;
        if let Some(reset_at) = next_reset {
            if reset_at <= now_rfc {
                let reset_id = Uuid::new_v4().to_string();
                tx.execute(
                    "INSERT INTO budget_reset_logs (id, user_id, budget_id, previous_spend, reset_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![reset_id, user_id, budget_id, spend, now_rfc],
                )?;
                current_spend = 0.0;
                let next = (now + chrono::Duration::seconds(duration_sec)).to_rfc3339();
                tx.execute(
                    "UPDATE users SET spend = 0.0, next_budget_reset_at = ?1 WHERE id = ?2",
                    params![next, user_id],
                )?;
            }
        } else {
            let next = (now + chrono::Duration::seconds(duration_sec)).to_rfc3339();
            tx.execute(
                "UPDATE users SET next_budget_reset_at = ?1 WHERE id = ?2",
                params![next, user_id],
            )?;
        }

        tx.commit()?;
        Ok((enforce, max_budget, current_spend))
    }

    /// Atomically record usage and increment user spend.
    pub fn record_usage(
        &self,
        key_id: Option<&str>,
        user_id: &str,
        model: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
        cost_usd: f64,
        request_id: &str,
    ) -> GatewayResult<()> {
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        insert_usage_tx(
            &tx,
            key_id,
            user_id,
            model,
            prompt_tokens,
            completion_tokens,
            cost_usd,
            request_id,
        )?;
        tx.execute(
            "UPDATE users SET spend = spend + ?1 WHERE id = ?2",
            params![cost_usd, user_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn list_usage(
        &self,
        user_id: Option<&str>,
        key_id: Option<&str>,
        limit: u32,
    ) -> GatewayResult<Vec<UsageRecord>> {
        let conn = self.lock()?;
        let limit = limit.clamp(1, 1000);
        let mut sql = String::from(
            "SELECT id, key_id, user_id, model, prompt_tokens, completion_tokens, cost_usd, request_id, created_at FROM usage_logs WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(uid) = user_id {
            sql.push_str(" AND user_id = ?");
            param_values.push(Box::new(uid.to_string()));
        }
        if let Some(kid) = key_id {
            sql.push_str(" AND key_id = ?");
            param_values.push(Box::new(kid.to_string()));
        }
        sql.push_str(" ORDER BY created_at DESC LIMIT ?");
        param_values.push(Box::new(i64::from(limit)));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), map_usage)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Delete usage rows whose stored cost is exactly `$0.00`.
    ///
    /// Those rows are leftover local estimates for models with no price table.
    /// Spend is unchanged because those inserts added `0` to `users.spend`.
    pub fn delete_zero_cost_usage(&self) -> GatewayResult<u64> {
        let conn = self.lock()?;
        let deleted = conn.execute("DELETE FROM usage_logs WHERE cost_usd = 0", [])?;
        Ok(deleted as u64)
    }

    pub fn usage_summary(
        &self,
        user_id: Option<&str>,
        key_id: Option<&str>,
        from: Option<&str>,
        to: Option<&str>,
        group_by: UsageSummaryGroupBy,
    ) -> GatewayResult<UsageSummary> {
        let conn = self.lock()?;
        let mut where_sql = String::from(" WHERE 1=1");
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(uid) = user_id {
            where_sql.push_str(" AND user_id = ?");
            params.push(Box::new(uid.to_string()));
        }
        if let Some(kid) = key_id {
            where_sql.push_str(" AND key_id = ?");
            params.push(Box::new(kid.to_string()));
        }
        if let Some(from) = from {
            where_sql.push_str(" AND created_at >= ?");
            params.push(Box::new(from.to_string()));
        }
        if let Some(to) = to {
            where_sql.push_str(" AND created_at <= ?");
            params.push(Box::new(to.to_string()));
        }

        let totals_sql = format!(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost_usd), 0.0) FROM usage_logs{where_sql}"
        );
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        let (requests, prompt_tokens, completion_tokens, cost_usd): (i64, i64, i64, f64) = conn
            .query_row(&totals_sql, params_ref.as_slice(), |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;

        let group_col = group_by.group_sql();
        let groups_sql = format!(
            "SELECT {group_col}, COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost_usd), 0.0) FROM usage_logs{where_sql} GROUP BY {group_col} ORDER BY SUM(cost_usd) DESC"
        );
        let mut stmt = conn.prepare(&groups_sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), |row| {
            Ok(UsageSummaryRow {
                key: row.get(0)?,
                requests: row.get::<_, i64>(1)? as u64,
                prompt_tokens: row.get::<_, i64>(2)? as u64,
                completion_tokens: row.get::<_, i64>(3)? as u64,
                cost_usd: row.get(4)?,
            })
        })?;
        let groups = rows.collect::<Result<Vec<_>, _>>()?;

        Ok(UsageSummary {
            totals: UsageSummaryTotals {
                requests: requests as u64,
                prompt_tokens: prompt_tokens as u64,
                completion_tokens: completion_tokens as u64,
                cost_usd,
            },
            groups,
        })
    }

    pub fn list_budget_reset_logs(
        &self,
        user_id: Option<&str>,
        limit: u32,
    ) -> GatewayResult<Vec<BudgetResetLogRecord>> {
        let conn = self.lock()?;
        let limit = limit.clamp(1, 1000);
        let mut sql = String::from(
            "SELECT id, user_id, budget_id, previous_spend, reset_at FROM budget_reset_logs WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(uid) = user_id {
            sql.push_str(" AND user_id = ?");
            param_values.push(Box::new(uid.to_string()));
        }
        sql.push_str(" ORDER BY reset_at DESC LIMIT ?");
        param_values.push(Box::new(i64::from(limit)));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), |row| {
            Ok(BudgetResetLogRecord {
                id: row.get(0)?,
                user_id: row.get(1)?,
                budget_id: row.get(2)?,
                previous_spend: row.get(3)?,
                reset_at: row.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
}

fn run_migrations(conn: &Connection) -> GatewayResult<()> {
    let mut version: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version < 1 {
        conn.execute_batch(MIGRATION_V1)?;
        version = 1;
        conn.pragma_update(None, "user_version", version)?;
    }
    if version < 2 {
        conn.execute_batch(MIGRATION_V2)?;
        conn.pragma_update(None, "user_version", 2)?;
    }
    Ok(())
}

fn merge_metadata_json(existing: Option<&str>, patch: &serde_json::Value) -> serde_json::Value {
    let mut base = existing
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    if let (Some(base_obj), Some(patch_obj)) = (base.as_object_mut(), patch.as_object()) {
        for (k, v) in patch_obj {
            if v.is_null() {
                base_obj.remove(k);
            } else {
                base_obj.insert(k.clone(), v.clone());
            }
        }
        base
    } else {
        patch.clone()
    }
}

fn compute_next_reset(conn: &Connection, budget_id: &str) -> GatewayResult<String> {
    let duration_sec: i64 = conn.query_row(
        "SELECT duration_sec FROM budgets WHERE id = ?1",
        params![budget_id],
        |row| row.get(0),
    )?;
    Ok((chrono::Utc::now() + chrono::Duration::seconds(duration_sec)).to_rfc3339())
}

fn insert_usage_tx(
    tx: &Transaction<'_>,
    key_id: Option<&str>,
    user_id: &str,
    model: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
    cost_usd: f64,
    request_id: &str,
) -> GatewayResult<()> {
    let id = Uuid::new_v4().to_string();
    let created_at = chrono::Utc::now().to_rfc3339();
    tx.execute(
        "INSERT INTO usage_logs (id, key_id, user_id, model, prompt_tokens, completion_tokens, cost_usd, request_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![id, key_id, user_id, model, prompt_tokens, completion_tokens, cost_usd, request_id, created_at],
    )?;
    Ok(())
}

fn map_user(row: &rusqlite::Row<'_>) -> rusqlite::Result<UserRecord> {
    Ok(UserRecord {
        id: row.get(0)?,
        alias: row.get(1)?,
        budget_id: row.get(2)?,
        spend: row.get(3)?,
        next_budget_reset_at: row.get(4)?,
        created_at: row.get(5)?,
    })
}

fn map_budget(row: &rusqlite::Row<'_>) -> rusqlite::Result<BudgetRecord> {
    Ok(BudgetRecord {
        id: row.get(0)?,
        max_budget: row.get(1)?,
        duration_sec: row.get(2)?,
        enforce: row.get::<_, i32>(3)? != 0,
        created_at: row.get(4)?,
    })
}

fn map_api_key(row: &rusqlite::Row<'_>) -> rusqlite::Result<ApiKeyRecord> {
    Ok(ApiKeyRecord {
        id: row.get(0)?,
        key_prefix: row.get(1)?,
        name: row.get(2)?,
        user_id: row.get(3)?,
        active: row.get(4)?,
        expires_at: row.get(5)?,
        metadata_json: row.get(6)?,
        created_at: row.get(7)?,
    })
}

fn map_usage(row: &rusqlite::Row<'_>) -> rusqlite::Result<UsageRecord> {
    Ok(UsageRecord {
        id: row.get(0)?,
        key_id: row.get(1)?,
        user_id: row.get(2)?,
        model: row.get(3)?,
        prompt_tokens: row.get::<_, i32>(4)? as u32,
        completion_tokens: row.get::<_, i32>(5)? as u32,
        cost_usd: row.get(6)?,
        request_id: row.get(7)?,
        created_at: row.get(8)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_and_user_key_flow() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        db.create_user("user-1", Some("Alice"), None).unwrap();
        let key = db
            .create_api_key(Some("test"), "user-1", &["openai:*".into()], None, None)
            .unwrap();
        assert!(key.plaintext_key.starts_with("sgw-"));
        let auth = hash_key(&key.plaintext_key);
        let found = db.lookup_api_key_by_hash(&auth).unwrap().unwrap();
        assert_eq!(found.user_id, "user-1");
    }

    #[test]
    fn opens_with_concurrent_write_pragmas() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path().join("test.db")).unwrap();
        let conn = db.lock().unwrap();
        let busy_timeout: i64 = conn
            .pragma_query_value(None, "busy_timeout", |row| row.get(0))
            .unwrap();
        let synchronous: i64 = conn
            .pragma_query_value(None, "synchronous", |row| row.get(0))
            .unwrap();
        let temp_store: i64 = conn
            .pragma_query_value(None, "temp_store", |row| row.get(0))
            .unwrap();
        assert_eq!(busy_timeout, 5_000);
        assert_eq!(synchronous, 1);
        assert_eq!(temp_store, 2);
    }

    #[test]
    fn delete_user_removes_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path).unwrap();
        db.create_user("user-1", None, None).unwrap();
        db.create_api_key(Some("k1"), "user-1", &["openai:*".into()], None, None)
            .unwrap();
        db.create_api_key(Some("k2"), "user-1", &["anthropic:*".into()], None, None)
            .unwrap();
        assert_eq!(db.delete_user("user-1").unwrap(), 2);
        assert!(!db.user_exists("user-1").unwrap());
        assert!(db.list_api_keys().unwrap().is_empty());
    }

    #[test]
    fn prepare_user_budget_is_read_only_when_reset_is_not_due() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path().join("test.db")).unwrap();
        let budget = db.create_budget(10.0, 86_400, true).unwrap();
        db.create_user("user-1", None, Some(&budget.id)).unwrap();
        db.record_usage(None, "user-1", "openai:gpt-4o-mini", 10, 5, 0.25, "req-1")
            .unwrap();
        let before = db.get_user("user-1").unwrap().unwrap();
        let (enforce, max_budget, spend) = db.prepare_user_budget("user-1").unwrap();
        assert!(enforce);
        assert_eq!(max_budget, 10.0);
        assert_eq!(spend, 0.25);
        let after = db.get_user("user-1").unwrap().unwrap();
        assert_eq!(after.spend, before.spend);
        assert_eq!(after.next_budget_reset_at, before.next_budget_reset_at);
        assert!(
            db.list_budget_reset_logs(Some("user-1"), 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn prepare_user_budget_resets_when_window_is_past() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path().join("test.db")).unwrap();
        let budget = db.create_budget(10.0, 86_400, true).unwrap();
        db.create_user("user-1", None, Some(&budget.id)).unwrap();
        db.record_usage(None, "user-1", "openai:gpt-4o-mini", 10, 5, 0.25, "req-1")
            .unwrap();
        {
            let conn = db.lock().unwrap();
            conn.execute(
                "UPDATE users SET next_budget_reset_at = ?1 WHERE id = ?2",
                params!["2000-01-01T00:00:00+00:00", "user-1"],
            )
            .unwrap();
        }
        let (enforce, max_budget, spend) = db.prepare_user_budget("user-1").unwrap();
        assert!(enforce);
        assert_eq!(max_budget, 10.0);
        assert_eq!(spend, 0.0);
        let after = db.get_user("user-1").unwrap().unwrap();
        assert_eq!(after.spend, 0.0);
        assert_ne!(
            after.next_budget_reset_at.as_deref(),
            Some("2000-01-01T00:00:00+00:00")
        );
        assert_eq!(
            db.list_budget_reset_logs(Some("user-1"), 10).unwrap().len(),
            1
        );
    }
}
