use std::env;
use std::path::PathBuf;

fn main() {
    // Retrieve environment variables
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");

    // Construct the path to the build directory
    let build_dir = PathBuf::from(format!("{}/../../../", &out_dir));

    // Construct the path to the deps directory
    let deps_dir = build_dir.join("deps");

    // Add RPATH linker arguments
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/icicle/lib", deps_dir.display());

    // Add iOS-specific configuration
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "ios" {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }

        // ====================================================================
    // ** NEW ANDROID-AWARE LOGIC **
    // ====================================================================
    // if target_os == "android" {
    //     println!("cargo:warning=Configuring for Android cross-compilation");

    //     let ndk_home = env::var("ANDROID_NDK_HOME")
    //         .expect("ANDROID_NDK_HOME is not set. Please set it to your Android NDK root.");

    //     // let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    //     // let android_abi = match target_arch.as_str() {
    //     //     "aarch64" => "arm64-v8a",
    //     //     "arm" => "armeabi-v7a",
    //     //     "x86_64" => "x86_64",
    //     //     "x86" => "x86",
    //     //     _ => panic!("Unsupported Android architecture: {}", target_arch),
    //     // };

    //     // let toolchain_file = format!("{}/build/cmake/android.toolchain.cmake", ndk_home);

    //     config
    //         .define("BUILD_FOR_ANDROID", "ON")
    //         .define("ANDROID_NDK_HOME", ndk_home);
    //         // .define("CMAKE_TOOLCHAIN_FILE", toolchain_file);
    //         // .define("ANDROID_ABI", android_abi)
    // } else {
    //     println!("cargo:warning=Configuring for non-Android platform");
    // }
}
