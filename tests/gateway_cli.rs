//! CLI integration tests for `superglue gateway` manage commands.

#![cfg(feature = "gateway")]

use assert_cmd::Command;
use predicates::prelude::*;

fn superglue_cmd() -> Command {
    let mut cmd = Command::cargo_bin("superglue").unwrap();
    cmd.env_remove("SUPERGLUE_GATEWAY_URL")
        .env_remove("GATEWAY_MASTER_KEY");
    cmd
}

#[test]
fn gateway_help_lists_manage_commands() {
    superglue_cmd()
        .arg("gateway")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("user"))
        .stdout(predicate::str::contains("key"))
        .stdout(predicate::str::contains("budget"))
        .stdout(predicate::str::contains("usage"))
        .stdout(predicate::str::contains("model"));
}

#[test]
fn user_create_and_list_via_cli() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cli.db");

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "user",
            "create",
            "--user-id",
            "alice",
            "--alias",
            "Alice",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Created user alice"));

    superglue_cmd()
        .args(["gateway", "--db", db.to_str().unwrap(), "user", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("alice"));
}

#[test]
fn key_create_requires_user() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cli.db");

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "key",
            "create",
            "--user-id",
            "missing",
            "--model",
            "openai:*",
        ])
        .assert()
        .failure();
}

#[test]
fn key_create_and_list_via_cli() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cli.db");

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "user",
            "create",
            "--user-id",
            "bob",
        ])
        .assert()
        .success();

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "key",
            "create",
            "--user-id",
            "bob",
            "--model",
            "openai:gpt-4o-mini",
            "--model",
            "anthropic:*",
            "--name",
            "bob-app",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("sgw-"))
        .stdout(predicate::str::contains("shown once"));

    superglue_cmd()
        .args(["gateway", "--db", db.to_str().unwrap(), "key", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("bob"))
        .stdout(predicate::str::contains("openai:gpt-4o-mini"));
}

#[test]
fn user_delete_via_cli() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cli.db");

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "user",
            "create",
            "--user-id",
            "carol",
        ])
        .assert()
        .success();

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "key",
            "create",
            "--user-id",
            "carol",
            "--model",
            "openai:*",
        ])
        .assert()
        .success();

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "user",
            "delete",
            "--user-id",
            "carol",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Deleted user carol"))
        .stdout(predicate::str::contains("1 key(s) revoked"));

    superglue_cmd()
        .args(["gateway", "--db", db.to_str().unwrap(), "user", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("carol").not());
}

#[test]
fn budget_create_via_cli() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cli.db");

    superglue_cmd()
        .args([
            "gateway",
            "--db",
            db.to_str().unwrap(),
            "budget",
            "create",
            "--max-budget",
            "10",
            "--duration-sec",
            "2592000",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Created budget"));
}
