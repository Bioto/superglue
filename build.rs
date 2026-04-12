fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc not found");
    // SAFETY: build scripts run single-threaded before any user code.
    unsafe { std::env::set_var("PROTOC", protoc) };
    prost_build::compile_protos(&["proto/superglue.proto"], &["proto/"])
        .expect("prost codegen failed");
}
