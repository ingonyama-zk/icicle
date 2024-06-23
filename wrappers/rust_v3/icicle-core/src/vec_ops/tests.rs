#![allow(unused_imports)]
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::device::Device;
use icicle_runtime::{runtime, stream::IcicleStream};

#[test]
fn test_vec_ops_config() {
    let mut vec_ops_config = VecOpsConfig::default();
    vec_ops_config
        .ext
        .set_int("int_example", 5);

    assert_eq!(
        vec_ops_config
            .ext
            .get_int("int_example"),
        5
    );

    // just to test the stream can be set and used correctly
    let mut stream = IcicleStream::create().unwrap();
    vec_ops_config.stream_handle = *stream;

    stream
        .synchronize()
        .unwrap();
    stream
        .release()
        .unwrap();
}
