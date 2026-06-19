// Android 15+ 16KB page devices require LOAD segment alignment. See:
// https://developer.android.com/guide/practices/page-sizes
fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("android") {
        println!("cargo:rustc-link-arg=-Wl,-z,max-page-size=16384");
        println!("cargo:rustc-link-arg=-Wl,-z,common-page-size=16384");
    }
}
