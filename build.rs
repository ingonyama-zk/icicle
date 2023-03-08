fn main() {
    //TODO: check cargo features selected
    //TODO: can conflict/duplicate with make ?

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=./icicle");

    let mut nvcc = cc::Build::new();

    nvcc.cuda(true);
    nvcc.debug(false);
    nvcc.flag("-arch=native");
    nvcc.file("./icicle/lib.cu").compile("ingo_icicle"); //TODO: extension??

}
