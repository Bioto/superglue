use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

#[test]
fn help_prints_usage() {
    cargo_bin_cmd!("superglue")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Superglue CLI"));
}

#[test]
fn version_subcommand_prints_version() {
    let expected = env!("CARGO_PKG_VERSION");
    cargo_bin_cmd!("superglue")
        .args(["version"])
        .assert()
        .success()
        .stdout(predicate::str::contains(expected));
}

#[test]
fn hello_default() {
    cargo_bin_cmd!("superglue")
        .args(["hello"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Hello, world!"));
}

#[test]
fn hello_with_name() {
    cargo_bin_cmd!("superglue")
        .args(["hello", "Rust"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Hello, Rust!"));
}
