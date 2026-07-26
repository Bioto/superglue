mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, black_box as cblack_box, criterion_group, criterion_main};
use superglue::context::AaakCompressor;
use superglue::context::condense_tool_round;
use superglue::openai::{ChatMessage, FunctionCall, MessageContent, ToolCall};

fn multi_tool_round_messages() -> Vec<ChatMessage> {
    let assistant = ChatMessage {
        role: "assistant".into(),
        content: None,
        tool_calls: Some(vec![
            ToolCall {
                id: "c1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "fetch_metrics".into(),
                    arguments: r#"{"region":"us-east"}"#.into(),
                },
            },
            ToolCall {
                id: "c2".into(),
                function: FunctionCall {
                    name: "fetch_metrics".into(),
                    arguments: r#"{"region":"eu-west"}"#.into(),
                },
                kind: "function".into(),
            },
            ToolCall {
                id: "c3".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_config".into(),
                    arguments: "{}".into(),
                },
            },
        ]),
        tool_call_id: None,
        name: None,
        refusal: None,
    };
    vec![
        assistant,
        ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text(
                r#"{"latency_ms":42,"errors":847,"samples":[1,2,3]}"#.into(),
            )),
            tool_calls: None,
            tool_call_id: Some("c1".into()),
            name: Some("fetch_metrics".into()),
            refusal: None,
        },
        ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text(
                r#"{"latency_ms":55,"errors":12,"samples":[4,5,6]}"#.into(),
            )),
            tool_calls: None,
            tool_call_id: Some("c2".into()),
            name: Some("fetch_metrics".into()),
            refusal: None,
        },
        ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text(
                r#"{"pool_size":10,"timeout_ms":8500,"flags":{"HttpOnly":true}}"#.into(),
            )),
            tool_calls: None,
            tool_call_id: Some("c3".into()),
            name: Some("get_config".into()),
            refusal: None,
        },
    ]
}

fn bench_condense_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("condense_encoding");
    for aaak in [false, true] {
        let label = if aaak { "aaak" } else { "plain" };
        group.bench_with_input(
            BenchmarkId::new("condense_tool_round", label),
            &aaak,
            |b, &use_aaak| {
                b.iter(|| {
                    let mut messages = multi_tool_round_messages();
                    condense_tool_round(&mut messages, use_aaak);
                    cblack_box(messages.len());
                    cblack_box(
                        messages
                            .first()
                            .and_then(|m| m.content.as_ref())
                            .and_then(|c| c.as_text())
                            .map(str::len)
                            .unwrap_or(0),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_aaak_encode(c: &mut Criterion) {
    let messages = multi_tool_round_messages();
    let tool_calls = messages[0].tool_calls.clone().unwrap();
    let tool_msgs: Vec<ChatMessage> = messages[1..].to_vec();
    let mut id_to_name = std::collections::HashMap::new();
    for tc in &tool_calls {
        id_to_name.insert(tc.id.clone(), tc.function.name.clone());
    }

    c.bench_function("aaak_encode_tool_round", |b| {
        b.iter(|| {
            let out = AaakCompressor::encode_tool_round(
                black_box(&tool_calls),
                black_box(&tool_msgs),
                black_box(&id_to_name),
            );
            cblack_box(out.len());
        });
    });
}

criterion_group!(benches, bench_condense_encoding, bench_aaak_encode);
criterion_main!(benches);
