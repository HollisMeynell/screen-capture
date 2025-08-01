fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::path::Path::new(&crate_dir).join("target");
    let out_path = out_dir.join("screen_capture.h");

    cbindgen::generate(&crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file(&out_path);
}
