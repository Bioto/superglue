fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc not found");
    // SAFETY: build scripts run single-threaded before any user code.
    unsafe { std::env::set_var("PROTOC", protoc) };

    if std::env::var("CARGO_FEATURE_GRPC").is_ok() {
        // When the `grpc` feature is enabled, tonic_build generates both the prost message
        // types AND the service trait + server/client stubs in one pass.
        tonic_prost_build::configure()
            .build_server(true)
            .build_client(true)
            .compile_protos(&["proto/superglue.proto"], &["proto/"])
            .expect("tonic+prost codegen failed");
    } else {
        prost_build::compile_protos(&["proto/superglue.proto"], &["proto/"])
            .expect("prost codegen failed");
    }
}
