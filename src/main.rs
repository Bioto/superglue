use clap::{Parser, Subcommand};

use superglue::telemetry::{LogFormat, ScrubMode, TelemetryConfig, init_tracing};

#[derive(Parser)]
#[command(name = "superglue", version, about = "Superglue CLI")]
struct Cli {
    /// Log level filter (overridden by RUST_LOG env var).
    #[arg(long, default_value = "info", global = true)]
    log_level: String,

    /// OTLP endpoint for span export (e.g. http://localhost:4318).
    /// Requires the crate to be compiled with the `otlp` feature.
    #[arg(long, global = true)]
    otlp_endpoint: Option<String>,

    /// Sensitive field handling in log output: redact (default), hash, allow.
    ///
    /// - redact: replace values with [REDACTED] (safe for production).
    /// - hash:   replace values with [HASH:xxxxxxxx] (debug sessions).
    /// - allow:  print values verbatim (local development only).
    #[arg(long, global = true, default_value = "redact")]
    scrub_mode: String,

    /// Log output format: pretty (human-readable) or json (one JSON object per line).
    #[arg(long, global = true, value_enum, default_value_t = LogFormatArg::Pretty)]
    log_format: LogFormatArg,

    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum LogFormatArg {
    #[default]
    Pretty,
    Json,
}

impl From<LogFormatArg> for LogFormat {
    fn from(a: LogFormatArg) -> Self {
        match a {
            LogFormatArg::Pretty => LogFormat::Pretty,
            LogFormatArg::Json => LogFormat::Json,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Print crate version.
    Version,
    /// Say hello.
    Hello {
        /// Name to greet.
        #[arg(default_value = "world")]
        name: String,
    },
    /// Start the gRPC server (requires `--features grpc`).
    #[cfg(feature = "grpc")]
    Serve {
        /// Address to listen on.
        #[arg(long, default_value = "0.0.0.0:50051")]
        addr: String,

        /// API key forwarded to the LLM provider (can also be set via OPENAI_API_KEY).
        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: String,

        /// LLM provider base URL.
        #[arg(long, default_value = "https://api.openai.com")]
        base_url: String,

        /// Default model name.
        #[arg(long, default_value = "gpt-5.4-nano-2026-03-17-mini")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();

    let scrub_mode_parsed = ScrubMode::from_str(&cli.scrub_mode);
    let scrub_unknown = scrub_mode_parsed.is_none();
    let scrub_mode = scrub_mode_parsed.unwrap_or(ScrubMode::Redact);

    init_tracing(TelemetryConfig {
        otlp_endpoint: cli.otlp_endpoint.clone(),
        log_level: cli.log_level.clone(),
        scrub_mode,
        log_format: cli.log_format.into(),
    });

    if scrub_unknown {
        tracing::warn!(
            scrub_mode = %cli.scrub_mode,
            "unknown --scrub-mode; defaulted to redact"
        );
    }

    superglue::telemetry::metrics::init_metrics();

    match cli.command {
        Command::Version => {
            println!("{}", superglue::version());
        }
        Command::Hello { name } => {
            println!("{}", superglue::greet(&name));
        }
        #[cfg(feature = "grpc")]
        Command::Serve {
            addr,
            api_key,
            base_url,
            model,
        } => {
            use std::sync::Arc;
            use superglue::chat::ChatOptions;
            use superglue::guardrails::GuardrailRegistry;
            use superglue::hooks::HookRegistry;
            use superglue::http::{ClientConfig, HttpClient};
            use superglue::tools::ToolRegistry;

            let http = Arc::new(
                HttpClient::new(ClientConfig::default()).expect("failed to build HttpClient"),
            );
            let tools = Arc::new(ToolRegistry::new());
            let hooks = Arc::new(HookRegistry::new());
            let guardrails = Arc::new(GuardrailRegistry::new());

            // Validate that base_url + api_key + model are parseable.
            let _opts = ChatOptions::new(&base_url, &api_key, &model);

            if let Err(e) = superglue::grpc::serve(&addr, http, tools, hooks, guardrails).await {
                tracing::error!(error = %e, "gRPC server error");
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    std::process::ExitCode::SUCCESS
}
