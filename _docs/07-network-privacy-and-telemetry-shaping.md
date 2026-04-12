# Network privacy and telemetry shaping

## Purpose of this document

The design discussion included ideas for **obfuscating or shaping traffic** (padding, jitter, disguised endpoints, steganography, domain fronting). This page records those themes for completeness and separates them from **core product requirements**.

**Important**: several techniques can violate **terms of service** for CDNs or providers, create **compliance** problems, or harm **user trust** if used without disclosure. Treat this material as **engineering background and ethics-sensitive**, not a shipping checklist.

## TLS is necessary but not sufficient

TLS hides payload contents from passive observers on the wire in typical configurations. Metadata often remains visible:

- That a connection occurred.
- Rough timing and size patterns.
- Destination identities unless additional measures apply.

## Application-layer payload shaping (exploratory)

Ideas mentioned in discussion:

- **Encrypt-then-pad** payloads to fixed block sizes so message length does not reveal operation type.
- **Randomized intervals** with jitter instead of perfectly periodic heartbeats to reduce fingerprinting.

If implemented, these belong next to explicit **privacy policies** and **user consent** where applicable.

## Destination hiding (high risk)

**Domain fronting** and similar tricks can make traffic resemble benign destinations. Many CDNs prohibit this; legal and policy review is mandatory before any experiment.

**DNS over HTTPS** and rotating endpoints may improve privacy against local DNS observers; they do not replace TLS or application redaction.

## Traffic mimicry and steganography (research)

Embedding control signals in images or mimicking browser telemetry patterns is **interesting for learning** and **dangerous to ship casually**. It can violate norms, policies, and laws depending on jurisdiction and deployment.

## Relationship to core design

The **default** architecture should assume **honest, documented endpoints**, **mTLS where needed**, and **scrubbed observability** (see [Observability](05-observability.md)). Optional shaping belongs behind explicit build-time or runtime flags with governance review.

## See also

- [Security and threat model](06-security-and-threat-model.md) for tiered assurance and honest limits.
