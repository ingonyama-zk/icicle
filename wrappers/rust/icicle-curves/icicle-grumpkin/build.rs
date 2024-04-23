use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    let mut config = Config::new("../../../../icicle");
    config
        .define("CURVE", "grumpkin")
        .define("CMAKE_BUILD_TYPE", "Release");

    #[cfg(feature = "devmode")]
    config.define("DEVMODE", "ON");

    let out_dir = config
        .build_target("icicle_curve")
        .build();

    println!("cargo:rustc-link-search={}/build/lib", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_field_grumpkin");
    println!("cargo:rustc-link-lib=ingo_curve_grumpkin");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
