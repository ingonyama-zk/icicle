use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    #[cfg(not(feature = "g2"))]
    let g2_str = "";
    #[cfg(feature = "g2")]
    let g2_str = "ON";

    let out_dir = Config::new("../../../../icicle")
        .define("BUILD_TESTS", "OFF")
        .define("CURVE", "bn254")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("G2_DEFINED", g2_str)
        .build_target("icicle")
        .build();

    println!("cargo:rustc-link-search={}/build", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
