use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "superglue", version, about = "Superglue CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print crate version
    Version,
    /// Say hello
    Hello {
        /// Name to greet
        #[arg(default_value = "world")]
        name: String,
    },
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Version => {
            println!("{}", superglue::version());
        }
        Command::Hello { name } => {
            println!("{}", superglue::greet(&name));
        }
    }
    std::process::ExitCode::SUCCESS
}
