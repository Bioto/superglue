//! Unit tests for `superglue::audit` — RunStore and RunRecorder.

use std::sync::Arc;

use superglue::audit::{RunRecorder, RunStore};
use superglue::hooks::{HookContext, HookHandler, HookStage};
use superglue::proto;

// ---------------------------------------------------------------------------
// RunStore tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runstore_insert_and_get() {
    let store = RunStore::new(None);
    let record = proto::RunRecord {
        request_id: "req-1".to_string(),
        messages: vec![],
        hook_events: vec![],
        outcome: None,
        started_at_ms: 1000,
        finished_at_ms: 2000,
    };
    store.insert(record).await;

    let found = store.get("req-1").await;
    assert!(found.is_some());
    assert_eq!(found.unwrap().request_id, "req-1");
}

#[tokio::test]
async fn runstore_get_missing_returns_none() {
    let store = RunStore::new(None);
    assert!(store.get("nonexistent").await.is_none());
}

#[tokio::test]
async fn runstore_list_all() {
    let store = RunStore::new(None);
    for i in 0..3 {
        store
            .insert(proto::RunRecord {
                request_id: format!("req-{i}"),
                messages: vec![],
                hook_events: vec![],
                outcome: None,
                started_at_ms: i as i64,
                finished_at_ms: i as i64 + 1,
            })
            .await;
    }
    let all = store.list().await;
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].request_id, "req-0");
    assert_eq!(all[2].request_id, "req-2");
}

#[tokio::test]
async fn runstore_clear() {
    let store = RunStore::new(None);
    store
        .insert(proto::RunRecord {
            request_id: "r".to_string(),
            messages: vec![],
            hook_events: vec![],
            outcome: None,
            started_at_ms: 0,
            finished_at_ms: 1,
        })
        .await;
    assert_eq!(store.len().await, 1);
    store.clear().await;
    assert_eq!(store.len().await, 0);
    assert!(store.is_empty().await);
}

#[tokio::test]
async fn runstore_capacity_cap_evicts_oldest() {
    let store = RunStore::new(Some(3));
    for i in 0..5u32 {
        store
            .insert(proto::RunRecord {
                request_id: format!("req-{i}"),
                messages: vec![],
                hook_events: vec![],
                outcome: None,
                started_at_ms: i as i64,
                finished_at_ms: i as i64 + 1,
            })
            .await;
    }
    // Only the 3 most recently inserted records should remain.
    assert_eq!(store.len().await, 3);
    let all = store.list().await;
    let ids: Vec<&str> = all.iter().map(|r| r.request_id.as_str()).collect();
    assert!(!ids.contains(&"req-0"), "req-0 should have been evicted");
    assert!(!ids.contains(&"req-1"), "req-1 should have been evicted");
    assert!(ids.contains(&"req-4"), "req-4 should be present");
}

// ---------------------------------------------------------------------------
// RunRecorder hook-handler tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runrecorder_accumulates_events_on_execute() {
    let store = Arc::new(RunStore::new(None));
    let recorder = RunRecorder::new(Arc::clone(&store));

    // Simulate two hook calls.
    let ctx1 = HookContext::new(HookStage::PreCompletion, "hello");
    let ctx2 = HookContext::with_meta(HookStage::PreTool, "{}", "tool_name", "echo");

    recorder.execute(ctx1).await.unwrap();
    recorder.execute(ctx2).await.unwrap();

    let events = recorder.peek_events().await;
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].stage, "pre_completion");
    assert_eq!(events[1].stage, "pre_tool");
    assert_eq!(events[1].metadata.get("tool_name").map(|s| s.as_str()), Some("echo"));
}

#[tokio::test]
async fn runrecorder_finish_stores_record_and_clears_events() {
    let store = Arc::new(RunStore::new(None));
    let recorder = RunRecorder::new(Arc::clone(&store));

    recorder
        .execute(HookContext::new(HookStage::PostCompletion, "hi"))
        .await
        .unwrap();

    let outcome = proto::CompletionOutcome {
        content: Some("hi".into()),
        rounds: 1,
        usage: None,
        finish_reason: Some("stop".into()),
    };

    recorder
        .finish("req-abc".to_string(), vec![], outcome, 1000, 2000)
        .await;

    // The record should now be in the store.
    let record = store.get("req-abc").await.unwrap();
    assert_eq!(record.request_id, "req-abc");
    assert_eq!(record.hook_events.len(), 1);
    assert_eq!(record.hook_events[0].stage, "post_completion");
    assert_eq!(record.started_at_ms, 1000);
    assert_eq!(record.finished_at_ms, 2000);

    // The recorder should have cleared its internal buffer after flushing.
    assert!(recorder.peek_events().await.is_empty(), "events not cleared after finish");
}

#[tokio::test]
async fn runrecorder_all_stages_returns_seven_variants() {
    let stages = RunRecorder::all_stages();
    assert_eq!(stages.len(), 7);
}

#[tokio::test]
async fn runrecorder_hook_events_include_timestamp() {
    let store = Arc::new(RunStore::new(None));
    let recorder = RunRecorder::new(Arc::clone(&store));

    recorder
        .execute(HookContext::new(HookStage::OnRetry, "retry"))
        .await
        .unwrap();

    let events = recorder.peek_events().await;
    assert!(events[0].timestamp_ms > 0, "timestamp should be positive");
}
