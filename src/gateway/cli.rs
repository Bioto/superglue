//! CLI for managing the gateway SQLite database (users, keys, budgets, usage).

use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde::Serialize;

use crate::gateway::db::Database;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::{serve, GatewayConfig};

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OutputFormat {
    #[default]
    Pretty,
    Json,
}

#[derive(Args, Clone, Debug)]
pub struct DbArgs {
    /// Path to the gateway SQLite database.
    #[arg(long, default_value = "superglue-gateway.db", global = true)]
    pub db: PathBuf,
}

#[derive(Subcommand, Clone, Debug)]
pub enum GatewayCommand {
    /// Start the HTTP LLM gateway server.
    Serve {
        /// Address to listen on.
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,

        /// Path to the SQLite database file.
        #[arg(long, default_value = "superglue-gateway.db")]
        db: String,

        /// Master key for admin operations.
        #[arg(long, env = "GATEWAY_MASTER_KEY")]
        master_key: String,
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
            db,
            master_key,
        } => {
            let config = GatewayConfig::new(addr, db.into(), master_key);
            serve(config).await
        }
        GatewayCommand::User { command } => run_user(&db_args.db, command),
        GatewayCommand::Key { command } => run_key(&db_args.db, command),
        GatewayCommand::Budget { command } => run_budget(&db_args.db, command),
        GatewayCommand::Usage { command } => run_usage(&db_args.db, command),
    }
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
            let key = db.update_api_key(&id, active, models_update, expires_update)?;
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

fn print_value<T: Serialize>(value: &T, format: OutputFormat, pretty: impl FnOnce(&T)) {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(value).expect("serialize"));
        }
        OutputFormat::Pretty => pretty(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
