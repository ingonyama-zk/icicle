use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    let out_dir = Config::new("../../../../icicle")
                .define("BUILD_TESTS", "OFF") //TODO: feature
                .define("CURVE", "grumpkin")
                .define("CMAKE_BUILD_TYPE", "Release")
                .build_target("icicle")
                .build();

    println!("cargo:rustc-link-search={}/build", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_grumpkin");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
