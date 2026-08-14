//! CLI for managing the gateway SQLite database (users, keys, budgets, usage)
//! or a remote gateway admin API over HTTPS.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde::Serialize;

use crate::gateway::db::Database;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::remote::RemoteClient;
use crate::gateway::{GatewayConfig, serve};

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OutputFormat {
    #[default]
    Pretty,
    Json,
}

#[derive(Args, Clone, Debug)]
pub struct DbArgs {
    /// Remote gateway base URL (e.g. https://gateway.example.com).
    /// When set, admin commands use the HTTP API instead of a local SQLite file.
    #[arg(long, env = "SUPERGLUE_GATEWAY_URL")]
    pub url: Option<String>,

    /// Master key for remote admin API calls.
    #[arg(long, env = "GATEWAY_MASTER_KEY")]
    pub master_key: Option<String>,

    /// Path to the gateway SQLite database (local mode only).
    #[arg(long, default_value = "superglue-gateway.db", global = true)]
    pub db: PathBuf,
}

impl DbArgs {
    fn remote_client(&self) -> GatewayResult<RemoteClient> {
        let url = self
            .url
            .clone()
            .filter(|u| !u.is_empty())
            .or_else(|| operator_env_value("SUPERGLUE_GATEWAY_URL"))
            .ok_or_else(|| GatewayError::Internal("remote URL not configured".into()))?;
        let master_key = self
            .master_key
            .clone()
            .filter(|k| !k.is_empty())
            .or_else(|| std::env::var("GATEWAY_MASTER_KEY").ok())
            .or_else(|| operator_env_value("GATEWAY_MASTER_KEY"))
            .filter(|k| !k.is_empty())
            .ok_or_else(|| {
                GatewayError::Internal(
                    "master key is required for remote gateway commands (set GATEWAY_MASTER_KEY or source ~/.superglue/gateway.env)".into(),
                )
            })?;
        RemoteClient::new(&url, &master_key)
    }
}

/// Read a variable from `~/.superglue/gateway.env` (works after `source` without `export`).
fn operator_env_value(name: &str) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home).join(".superglue/gateway.env");
    let contents = std::fs::read_to_string(path).ok()?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let (key, value) = line.split_once('=')?;
        if key.trim() == name {
            let value = value.trim();
            let value = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                .unwrap_or(value);
            return Some(value.to_string());
        }
    }
    None
}

#[derive(Subcommand, Clone, Debug)]
pub enum GatewayCommand {
    /// Start the HTTP LLM gateway server.
    Serve {
        /// Address to listen on.
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,

        /// Master key for admin operations.
        #[arg(long, env = "GATEWAY_MASTER_KEY")]
        master_key: String,

        /// S3 bucket for optional LLM traffic capture (requires `capture` feature).
        #[cfg(feature = "capture")]
        #[arg(long, env = "SUPERGLUE_CAPTURE_S3_BUCKET")]
        capture_s3_bucket: Option<String>,

        /// S3 key prefix for capture objects (default: gateway-capture).
        #[cfg(feature = "capture")]
        #[arg(long, env = "SUPERGLUE_CAPTURE_S3_PREFIX")]
        capture_s3_prefix: Option<String>,
    },
    /// Manage gateway users.
    User {
        #[command(subcommand)]
        command: UserCommand,
    },
    /// Manage virtual API keys and model allowlists.
    Key {
        #[command(subcommand)]
        command: KeyCommand,
    },
    /// Manage budget tiers.
    Budget {
        #[command(subcommand)]
        command: BudgetCommand,
    },
    /// View usage logs.
    Usage {
        #[command(subcommand)]
        command: UsageCommand,
    },
    /// List models available to an API key.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Subcommand, Clone, Debug)]
pub enum UserCommand {
    /// Create a user.
    Create {
        /// Unique user identifier.
        #[arg(long)]
        user_id: String,
        /// Display name.
        #[arg(long)]
        alias: Option<String>,
        /// Budget tier to assign.
        #[arg(long)]
        budget_id: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// List all users.
    List {
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// Update a user.
    Update {
        #[arg(long)]
        user_id: String,
        #[arg(long)]
        alias: Option<String>,
        #[arg(long)]
        budget_id: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// Delete a user and revoke all of their API keys.
    Delete {
        #[arg(long)]
        user_id: String,
    },
}

#[derive(Subcommand, Clone, Debug)]
pub enum KeyCommand {
    /// Create a virtual API key (plaintext shown once).
    Create {
        /// User this key belongs to.
        #[arg(long)]
        user_id: String,
        /// Allowed model pattern (repeatable, e.g. `openai:*`, `anthropic:claude-3`).
        #[arg(long = "model", required = true)]
        models: Vec<String>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        expires_at: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// List virtual keys (never shows full secrets).
    List {
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// Update a virtual key.
    Update {
        #[arg(long)]
        id: String,
        #[arg(long)]
        active: Option<bool>,
        /// Replace allowed model patterns (repeatable).
        #[arg(long = "model")]
        models: Vec<String>,
        #[arg(long)]
        expires_at: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// Revoke (delete) a virtual key.
    Delete {
        #[arg(long)]
        id: String,
    },
}

#[derive(Subcommand, Clone, Debug)]
pub enum BudgetCommand {
    /// Create a budget tier.
    Create {
        #[arg(long)]
        max_budget: f64,
        #[arg(long)]
        duration_sec: i64,
        /// Reject requests when spend exceeds the budget.
        #[arg(long, default_value_t = true)]
        enforce: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// List budget tiers.
    List {
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// Update a budget tier.
    Update {
        #[arg(long)]
        id: String,
        #[arg(long)]
        max_budget: Option<f64>,
        #[arg(long)]
        duration_sec: Option<i64>,
        #[arg(long)]
        enforce: Option<bool>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
    /// Delete a budget tier (clears budget_id on assigned users).
    Delete {
        #[arg(long)]
        id: String,
    },
}

#[derive(Subcommand, Clone, Debug)]
pub enum ModelCommand {
    /// List model patterns visible to an API key (master key or virtual key).
    List {
        /// API key to query (defaults to master key from env or ~/.superglue/gateway.env).
        #[arg(long)]
        key: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
}

#[derive(Subcommand, Clone, Debug)]
pub enum UsageCommand {
    /// List usage log entries.
    List {
        #[arg(long)]
        user_id: Option<String>,
        #[arg(long)]
        key_id: Option<String>,
        #[arg(long, default_value_t = 100)]
        limit: u32,
        #[arg(long, value_enum, default_value_t = OutputFormat::Pretty)]
        output: OutputFormat,
    },
}

#[derive(Serialize)]
struct KeyCreateOutput {
    id: String,
    key: String,
    key_prefix: String,
    user_id: String,
    allowed_models: Vec<String>,
}

/// Run a gateway CLI command.
pub async fn execute(db_args: &DbArgs, command: GatewayCommand) -> Result<(), GatewayError> {
    match command {
        GatewayCommand::Serve {
            addr,
            master_key,
            #[cfg(feature = "capture")]
            capture_s3_bucket,
            #[cfg(feature = "capture")]
            capture_s3_prefix,
        } => {
            let mut config = GatewayConfig::new(addr, db_args.db.clone(), master_key);
            #[cfg(feature = "capture")]
            {
                config.capture = crate::gateway::capture::CaptureConfig::from_serve_args(
                    &db_args.db,
                    capture_s3_bucket,
                    capture_s3_prefix,
                );
            }
            serve(config).await
        }
        GatewayCommand::User { command } => {
            if db_args.url.is_some() {
                run_user_remote(db_args, command).await
            } else {
                run_user(&db_args.db, command)
            }
        }
        GatewayCommand::Key { command } => {
            if db_args.url.is_some() {
                run_key_remote(db_args, command).await
            } else {
                run_key(&db_args.db, command)
            }
        }
        GatewayCommand::Budget { command } => {
            if db_args.url.is_some() {
                run_budget_remote(db_args, command).await
            } else {
                run_budget(&db_args.db, command)
            }
        }
        GatewayCommand::Usage { command } => {
            if db_args.url.is_some() {
                run_usage_remote(db_args, command).await
            } else {
                run_usage(&db_args.db, command)
            }
        }
        GatewayCommand::Model { command } => {
            if model_use_remote(db_args) {
                run_model_remote(db_args, command).await
            } else {
                run_model_local(db_args, command).await
            }
        }
    }
}

fn model_use_remote(db_args: &DbArgs) -> bool {
    db_args.url.as_ref().is_some_and(|u| !u.is_empty())
        || operator_env_value("SUPERGLUE_GATEWAY_URL").is_some()
}

fn resolve_list_key(db_args: &DbArgs, key: Option<String>) -> GatewayResult<String> {
    key.filter(|k| !k.is_empty())
        .or_else(|| db_args.master_key.clone())
        .or_else(|| std::env::var("GATEWAY_MASTER_KEY").ok())
        .or_else(|| operator_env_value("GATEWAY_MASTER_KEY"))
        .filter(|k| !k.is_empty())
        .ok_or_else(|| {
            GatewayError::Internal(
                "API key required for model list (--key or GATEWAY_MASTER_KEY)".into(),
            )
        })
}

fn master_key_hash(db_args: &DbArgs) -> GatewayResult<[u8; 32]> {
    let master = db_args
        .master_key
        .clone()
        .or_else(|| std::env::var("GATEWAY_MASTER_KEY").ok())
        .or_else(|| operator_env_value("GATEWAY_MASTER_KEY"))
        .filter(|k| !k.is_empty())
        .ok_or_else(|| {
            GatewayError::Internal(
                "master key required for local model list (GATEWAY_MASTER_KEY)".into(),
            )
        })?;
    Ok(crate::gateway::auth::hash_key(&master))
}

fn open_db(path: &PathBuf) -> GatewayResult<Database> {
    Database::open(path)
}

fn run_user(path: &PathBuf, command: UserCommand) -> GatewayResult<()> {
    let db = open_db(path)?;
    match command {
        UserCommand::Create {
            user_id,
            alias,
            budget_id,
            output,
        } => {
            let user = db.create_user(&user_id, alias.as_deref(), budget_id.as_deref())?;
            print_value(&user, output, |u| {
                println!("Created user {} (spend ${:.4})", u.id, u.spend);
                if let Some(a) = &u.alias {
                    println!("  alias: {a}");
                }
                if let Some(b) = &u.budget_id {
                    println!("  budget_id: {b}");
                }
            });
        }
        UserCommand::List { output } => {
            let users = db.list_users()?;
            print_value(&users, output, |users| {
                if users.is_empty() {
                    println!("No users.");
                    return;
                }
                for u in users {
                    println!(
                        "{}  spend=${:.4}  budget={}  alias={}",
                        u.id,
                        u.spend,
                        u.budget_id.as_deref().unwrap_or("-"),
                        u.alias.as_deref().unwrap_or("-"),
                    );
                }
            });
        }
        UserCommand::Update {
            user_id,
            alias,
            budget_id,
            output,
        } => {
            let alias_update = alias.as_ref().map(|a| Some(a.as_str()));
            let budget_update = budget_id.as_ref().map(|b| Some(b.as_str()));
            let user = db.update_user(&user_id, alias_update, budget_update)?;
            print_value(&user, output, |u| {
                println!("Updated user {}", u.id);
            });
        }
        UserCommand::Delete { user_id } => {
            let keys_deleted = db.delete_user(&user_id)?;
            println!("Deleted user {user_id} ({keys_deleted} key(s) revoked)");
        }
    }
    Ok(())
}

fn run_key(path: &PathBuf, command: KeyCommand) -> GatewayResult<()> {
    let db = open_db(path)?;
    match command {
        KeyCommand::Create {
            user_id,
            models,
            name,
            expires_at,
            output,
        } => {
            let result = db.create_api_key(
                name.as_deref(),
                &user_id,
                &models,
                expires_at.as_deref(),
                None,
            )?;
            let out = KeyCreateOutput {
                id: result.id.clone(),
                key: result.plaintext_key.clone(),
                key_prefix: result.key_prefix.clone(),
                user_id: user_id.clone(),
                allowed_models: models.clone(),
            };
            print_value(&out, output, |o| {
                println!("Created API key (save the key — shown once):");
                println!("  id: {}", o.id);
                println!("  key: {}", o.key);
                println!("  prefix: {}", o.key_prefix);
                println!("  user_id: {}", o.user_id);
                println!("  models: {}", o.allowed_models.join(", "));
            });
        }
        KeyCommand::List { output } => {
            let keys = db.list_api_keys()?;
            print_value(&keys, output, |keys| {
                if keys.is_empty() {
                    println!("No API keys.");
                    return;
                }
                for k in keys {
                    println!(
                        "{}  {}  user={}  active={}  models=[{}]",
                        k.id,
                        k.key_prefix,
                        k.user_id,
                        k.active,
                        k.allowed_models.join(", "),
                    );
                }
            });
        }
        KeyCommand::Update {
            id,
            active,
            models,
            expires_at,
            output,
        } => {
            let models_update = if models.is_empty() {
                None
            } else {
                Some(models.as_slice())
            };
            let expires_update = expires_at.as_ref().map(|e| Some(e.as_str()));
            let key = db.update_api_key(&id, active, models_update, expires_update, None)?;
            print_value(&key, output, |k| {
                println!("Updated key {} ({})", k.id, k.key_prefix);
            });
        }
        KeyCommand::Delete { id } => {
            db.delete_api_key(&id)?;
            println!("Deleted key {id}");
        }
    }
    Ok(())
}

fn run_budget(path: &PathBuf, command: BudgetCommand) -> GatewayResult<()> {
    let db = open_db(path)?;
    match command {
        BudgetCommand::Create {
            max_budget,
            duration_sec,
            enforce,
            output,
        } => {
            let budget = db.create_budget(max_budget, duration_sec, enforce)?;
            print_value(&budget, output, |b| {
                println!(
                    "Created budget {}  max=${:.2}  duration={}s  enforce={}",
                    b.id, b.max_budget, b.duration_sec, b.enforce
                );
            });
        }
        BudgetCommand::List { output } => {
            let budgets = db.list_budgets()?;
            print_value(&budgets, output, |budgets| {
                if budgets.is_empty() {
                    println!("No budgets.");
                    return;
                }
                for b in budgets {
                    println!(
                        "{}  max=${:.2}  duration={}s  enforce={}",
                        b.id, b.max_budget, b.duration_sec, b.enforce
                    );
                }
            });
        }
        BudgetCommand::Update {
            id,
            max_budget,
            duration_sec,
            enforce,
            output,
        } => {
            let budget = db.update_budget(&id, max_budget, duration_sec, enforce)?;
            print_value(&budget, output, |b| {
                println!(
                    "Updated budget {}  max=${:.2}  duration={}s  enforce={}",
                    b.id, b.max_budget, b.duration_sec, b.enforce
                );
            });
        }
        BudgetCommand::Delete { id } => {
            let users_cleared = db.delete_budget(&id)?;
            println!("Deleted budget {id} (cleared {users_cleared} user assignment(s))");
        }
    }
    Ok(())
}

fn run_usage(path: &PathBuf, command: UsageCommand) -> GatewayResult<()> {
    let db = open_db(path)?;
    match command {
        UsageCommand::List {
            user_id,
            key_id,
            limit,
            output,
        } => {
            let logs = db.list_usage(user_id.as_deref(), key_id.as_deref(), limit)?;
            print_value(&logs, output, |logs| {
                if logs.is_empty() {
                    println!("No usage logs.");
                    return;
                }
                for log in logs {
                    println!(
                        "{}  user={}  model={}  tokens={}/{}  cost=${:.4}  at={}",
                        log.id,
                        log.user_id,
                        log.model,
                        log.prompt_tokens,
                        log.completion_tokens,
                        log.cost_usd,
                        log.created_at,
                    );
                }
            });
        }
    }
    Ok(())
}

async fn run_model_local(db_args: &DbArgs, command: ModelCommand) -> GatewayResult<()> {
    match command {
        ModelCommand::List { key, output } => {
            let api_key = resolve_list_key(db_args, key)?;
            let db = open_db(&db_args.db)?;
            let auth =
                crate::gateway::auth::authenticate(&db, &api_key, &master_key_hash(db_args)?)?;
            let http = crate::http::HttpClient::new(crate::http::ClientConfig::default())
                .map_err(|e| GatewayError::Internal(e.to_string()))?;
            let credentials = crate::providers::ProviderCredentials::from_env();
            let models =
                crate::gateway::model_catalog::list_models(&http, &credentials, &auth).await?;
            print_models(&models, output);
        }
    }
    Ok(())
}

async fn run_model_remote(db_args: &DbArgs, command: ModelCommand) -> GatewayResult<()> {
    match command {
        ModelCommand::List { key, output } => {
            let client = db_args.remote_client()?;
            let api_key = resolve_list_key(db_args, key)?;
            let models = client.list_models(&api_key).await?;
            print_models(&models, output);
        }
    }
    Ok(())
}

fn print_models(models: &[serde_json::Value], format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(models).expect("serialize")
            );
        }
        OutputFormat::Pretty => {
            if models.is_empty() {
                println!("No models.");
                return;
            }
            for model in models {
                println!("{}", model["id"].as_str().unwrap_or("-"));
            }
        }
    }
}

fn print_value<T: Serialize>(value: &T, format: OutputFormat, pretty: impl FnOnce(&T)) {
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(value).expect("serialize")
            );
        }
        OutputFormat::Pretty => pretty(value),
    }
}

async fn run_user_remote(db_args: &DbArgs, command: UserCommand) -> GatewayResult<()> {
    let client = db_args.remote_client()?;
    match command {
        UserCommand::Create {
            user_id,
            alias,
            budget_id,
            output,
        } => {
            let user = client
                .create_user(&user_id, alias.as_deref(), budget_id.as_deref())
                .await?;
            print_value(&user, output, |u| {
                println!("Created user {} (spend ${:.4})", u.id, u.spend);
                if let Some(a) = &u.alias {
                    println!("  alias: {a}");
                }
                if let Some(b) = &u.budget_id {
                    println!("  budget_id: {b}");
                }
            });
        }
        UserCommand::List { output } => {
            let users = client.list_users().await?;
            print_value(&users, output, |users| {
                if users.is_empty() {
                    println!("No users.");
                    return;
                }
                for u in users {
                    println!(
                        "{}  spend=${:.4}  budget={}  alias={}",
                        u.id,
                        u.spend,
                        u.budget_id.as_deref().unwrap_or("-"),
                        u.alias.as_deref().unwrap_or("-"),
                    );
                }
            });
        }
        UserCommand::Update {
            user_id,
            alias,
            budget_id,
            output,
        } => {
            let user = client
                .update_user(
                    &user_id,
                    alias.as_deref(),
                    budget_id.as_ref().map(|b| Some(b.as_str())),
                )
                .await?;
            print_value(&user, output, |u| {
                println!("Updated user {}", u.id);
            });
        }
        UserCommand::Delete { user_id } => {
            let result = client.delete_user(&user_id).await?;
            let keys_deleted = result.keys_deleted;
            println!("Deleted user {user_id} ({keys_deleted} key(s) revoked)");
        }
    }
    Ok(())
}

async fn run_key_remote(db_args: &DbArgs, command: KeyCommand) -> GatewayResult<()> {
    let client = db_args.remote_client()?;
    match command {
        KeyCommand::Create {
            user_id,
            models,
            name,
            expires_at,
            output,
        } => {
            let created: serde_json::Value = client
                .create_key(&user_id, &models, name.as_deref(), expires_at.as_deref())
                .await?;
            print_value(&created, output, |v| {
                println!("Created API key (save the key — shown once):");
                println!("  id: {}", v["id"].as_str().unwrap_or("-"));
                println!("  key: {}", v["key"].as_str().unwrap_or("-"));
                println!("  prefix: {}", v["key_prefix"].as_str().unwrap_or("-"));
                println!("  user_id: {}", v["user_id"].as_str().unwrap_or("-"));
                if let Some(models) = v["allowed_models"].as_array() {
                    let joined: Vec<_> = models.iter().filter_map(|m| m.as_str()).collect();
                    println!("  models: {}", joined.join(", "));
                }
            });
        }
        KeyCommand::List { output } => {
            let keys = client.list_keys().await?;
            print_value(&keys, output, |keys| {
                if keys.is_empty() {
                    println!("No API keys.");
                    return;
                }
                for k in keys {
                    println!(
                        "{}  {}  user={}  active={}  models=[{}]",
                        k.id,
                        k.key_prefix,
                        k.user_id,
                        k.active,
                        k.allowed_models.join(", "),
                    );
                }
            });
        }
        KeyCommand::Update {
            id,
            active,
            models,
            expires_at,
            output,
        } => {
            let models_update = if models.is_empty() {
                None
            } else {
                Some(models.as_slice())
            };
            let key = client
                .update_key(
                    &id,
                    active,
                    models_update,
                    expires_at.as_ref().map(|e| Some(e.as_str())),
                    None,
                )
                .await?;
            print_value(&key, output, |k| {
                println!("Updated key {} ({})", k.id, k.key_prefix);
            });
        }
        KeyCommand::Delete { id } => {
            client.delete_key(&id).await?;
            println!("Deleted key {id}");
        }
    }
    Ok(())
}

async fn run_budget_remote(db_args: &DbArgs, command: BudgetCommand) -> GatewayResult<()> {
    let client = db_args.remote_client()?;
    match command {
        BudgetCommand::Create {
            max_budget,
            duration_sec,
            enforce,
            output,
        } => {
            let budget = client
                .create_budget(max_budget, duration_sec, enforce)
                .await?;
            print_value(&budget, output, |b| {
                println!(
                    "Created budget {}  max=${:.2}  duration={}s  enforce={}",
                    b.id, b.max_budget, b.duration_sec, b.enforce
                );
            });
        }
        BudgetCommand::List { output } => {
            let budgets = client.list_budgets().await?;
            print_value(&budgets, output, |budgets| {
                if budgets.is_empty() {
                    println!("No budgets.");
                    return;
                }
                for b in budgets {
                    println!(
                        "{}  max=${:.2}  duration={}s  enforce={}",
                        b.id, b.max_budget, b.duration_sec, b.enforce
                    );
                }
            });
        }
        BudgetCommand::Update {
            id,
            max_budget,
            duration_sec,
            enforce,
            output,
        } => {
            let budget = client
                .update_budget(&id, max_budget, duration_sec, enforce)
                .await?;
            print_value(&budget, output, |b| {
                println!(
                    "Updated budget {}  max=${:.2}  duration={}s  enforce={}",
                    b.id, b.max_budget, b.duration_sec, b.enforce
                );
            });
        }
        BudgetCommand::Delete { id } => {
            let result = client.delete_budget(&id).await?;
            print_value(&result, OutputFormat::Pretty, |r| {
                let users_cleared = r.get("users_cleared").and_then(|v| v.as_u64()).unwrap_or(0);
                println!("Deleted budget {id} (cleared {users_cleared} user assignment(s))");
            });
        }
    }
    Ok(())
}

async fn run_usage_remote(db_args: &DbArgs, command: UsageCommand) -> GatewayResult<()> {
    let client = db_args.remote_client()?;
    match command {
        UsageCommand::List {
            user_id,
            key_id,
            limit,
            output,
        } => {
            let logs = client
                .list_usage(user_id.as_deref(), key_id.as_deref(), limit)
                .await?;
            print_value(&logs, output, |logs| {
                if logs.is_empty() {
                    println!("No usage logs.");
                    return;
                }
                for log in logs {
                    println!(
                        "{}  user={}  model={}  tokens={}/{}  cost=${:.4}  at={}",
                        log.id,
                        log.user_id,
                        log.model,
                        log.prompt_tokens,
                        log.completion_tokens,
                        log.cost_usd,
                        log.created_at,
                    );
                }
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn user_create_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("cli.db");
        let db = Database::open(&db_path).unwrap();
        db.create_user("alice", Some("Alice"), None).unwrap();
        let users = db.list_users().unwrap();
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].id, "alice");
    }
}
