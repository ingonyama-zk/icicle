use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle_v3/");

    // Base config
    let mut config = Config::new("../../../../icicle_v3/");
    config
        .define("FIELD", "babybear")
        .define("EXT_FIELD", "ON")
        .define("CMAKE_BUILD_TYPE", "Release");

    // Build
    let out_dir = config
        .build_target("all")
        .build();

    println!("cargo:rustc-link-search={}/build", out_dir.display());
    println!("cargo:rustc-link-lib=icicle_field_babybear");
    println!("cargo:rustc-link-lib=stdc++");

    // default backends dir
    // println!(
    //     "cargo:rustc-env=DEFAULT_BACKEND_INSTALL_DIR={}/build/backend",
    //     out_dir.display()
    // );
}
