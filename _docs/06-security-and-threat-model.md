# Security and threat model

This document tiers expectations. **Not every product needs the strongest tier**; mixing “research” ideas into default requirements creates confusion.

## Tier A: Normal commercial packaging

Typical goals: protect proprietary orchestration logic, ship updates safely, use standard transport security.

- Ship the core as a **binary** or managed service; expose a **documented wire protocol** (for example gRPC with protobuf) rather than source.
- Use **TLS** for all remote traffic; prefer **mTLS** between clients and a local daemon when authentication matters.
- **Strip symbols** and enable **LTO** in release builds; keep dependency inventories (`cargo auditable` or SBOM processes) for supply-chain review.
- **License checks** (signed keys, online entitlement) belong here as product decisions, not as Rust language features.

## Tier B: Strong local confidentiality

Goals: reduce persistence of sensitive prompts and tool outputs; limit exposure from crash dumps and swap.

- Prefer **in-memory** workflow state where policy allows; avoid writing secrets to disk.
- Use **`zeroize`** on sensitive buffers and consider the **`secrecy`** pattern for key material.
- Optional **`mlock`** on particularly sensitive pages reduces swap risk but does not defeat all attacks.
- **gRPC with rustls**, short-lived certificates, and tight cipher configuration for any remote control plane traffic.

**Honest ceiling**: data sent to external LLM APIs **leaves the trust boundary**. Tooling must treat provider calls and logging as primary exfiltration risks. Memory hardening does not fix network egress or careless logs.

## Tier C: Research and adversarial topics

These items appeared in brainstorming (self-destruct flows, binary overwrite, remote heartbeats). Treat them as **exploratory** unless a product explicitly adopts them.

### Self-destruct and tamper response

Concepts include: overwrite executable bytes before deletion, zero sensitive state, abort on integrity failure, or react to remote “destruct” signals.

**Platform reality**: deleting or overwriting the **currently running executable** fails on some operating systems (notably Windows file locking). Any design must account for OS-specific launchers or delayed cleanup.

**Copy problem**: if an attacker copies the binary before a trigger, local deletion does not revoke that copy. Remote attestation or encrypted loaders shift the threat model but add operational cost.

### Encrypted loader model

At-rest binary encryption with keys delivered only after authentication can reduce casual copying. This is a **distribution architecture**, not a Rust language feature, and implies robust key management and availability tradeoffs.

## Observability as a leak vector

The largest practical leaks are often:

- **LLM API traffic** to third parties.
- **Tool outputs** containing secrets.
- **Tracing and logging** capturing full prompts or results.

Cross-reference [Observability](05-observability.md): redaction is mandatory for serious deployments.

## What “leave no trace” cannot promise in userspace

A determined attacker with **kernel or hypervisor access**, **physical memory access**, or **full disk forensics** may recover data despite zeroization. Claims should stay proportional to the threat model.

## See also

- [Network privacy and telemetry shaping](07-network-privacy-and-telemetry-shaping.md) for optional network-layer topics.
- [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md) for secure transport between clients and core.
