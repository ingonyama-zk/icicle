use cmake::Config;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=../../../../icicle");

    // Base config
    let mut config = Config::new("../../../../icicle/");
    config
        .define("CURVE", "bn254")
        .define("CMAKE_BUILD_TYPE", "Release");

    // Optional Features
    #[cfg(feature = "g2")]
    config.define("G2", "ON");

    #[cfg(feature = "ec_ntt")]
    config.define("ECNTT", "ON");

    #[cfg(feature = "devmode")]
    config.define("DEVMODE", "ON");

    if let Ok(cuda_arch) = env::var("CUDA_ARCH") {
        config.define("CUDA_ARCH", Some(&cuda_arch));
    }

    // Build
    let out_dir = config
        .build_target("icicle_curve")
        .build();

    println!("cargo:rustc-link-search={}/build/lib/", out_dir.display());

    println!("cargo:rustc-link-lib=ingo_field_bn254");
    println!("cargo:rustc-link-lib=ingo_curve_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
