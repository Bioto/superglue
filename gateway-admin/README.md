# SuperGlue gateway admin

This React application provides the operator UI for the SuperGlue gateway.

## Build

1. Run `npm ci`.
2. Run `npm run build`.
3. Build SuperGlue with `cargo build --features gateway`.
4. Open `/admin/` on the gateway.

The gateway embeds `dist/` into the binary. The checked-in `dist/index.html`
is a build hint for source checkouts. Run the build before deployment.

The UI keeps the gateway master key in `sessionStorage` and sends it to the
same-origin `/v1/*` admin API.
