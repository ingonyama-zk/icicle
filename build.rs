use cmake::Config;
use std::env::var;

fn main() {
    //TODO: check cargo features selected
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=./icicle");

    let cargo_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let profile = var("PROFILE").unwrap();

    let target_output_dir = format!("{}/target/{}", cargo_dir, profile);

    // let cuda_dir = PathBuf::from(
    //     env::var("CUDA_HOME")
    //       .or_else(|_| env::var("CUDA_PATH"))
    //       .unwrap_or_else(|_| "/usr/local/cuda".to_owned())
    // );
    // let cuda_include_dir = cuda_dir.join("include");
    // let cuda_lib_dir = cuda_dir.join("lib64");
    // println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    // let bindings = bindgen::Builder::default()
    // .clang_arg(format!("-I{}", cuda_include_dir.as_os_str().to_str().unwrap()))
    // .clang_arg(format!("-I{}", "icicle/appUtils/msm/"))
    // .header("icicle/appUtils/msm/msm.cuh")
    // .header("cuda_runtime_api.h")
    // .header("icicle/utils/device_context.cuh")
    // // .clang_arg("-x c++")
    // // .clang_arg("-std=c++17")
    // .rustfmt_bindings(true)
    // // Add any additional configuration here...
    // .generate()
    // .expect("Unable to generate bindings");

    // let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // bindings
    //     .write_to_file(out_path.join("bindings.rs"))
    //     .expect("Couldn't write bindings!");

    Config::new("./icicle")
                .define("BUILD_TESTS", "OFF") //TODO: feature
                // .define("CURVE", "12381")
                // .define("CURVE", "bls12_381")
                .define("CURVE", "bn254")
                .define("LIBRARY_OUTPUT_DIRECTORY", &target_output_dir)
                .build_target("icicle")
                .build();

    println!("cargo:rustc-link-search={}", &target_output_dir);

    // println!("cargo:rustc-link-lib=ingo_bls12_381");
    println!("cargo:rustc-link-lib=ingo_bn254");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
}
