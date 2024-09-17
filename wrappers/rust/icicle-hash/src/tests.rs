#[cfg(test)]
mod tests {

    use crate::{keccak, sha3};
    use icicle_core::hash::HashConfig;
    use icicle_runtime::memory::HostSlice;

    #[test]
    fn keccak_hashing() {
        let keccak_hasher = keccak::create_keccak_256_hasher(0).unwrap();
        let input = vec![0 as u8; 90];
        let mut output = vec![0 as u8; 96]; // 256b * batch
        keccak_hasher
            .hash(
                HostSlice::from_slice(&input),
                &HashConfig::default(),
                HostSlice::from_mut_slice(&mut output),
            )
            .unwrap();
        println!("output= {:?}", output);
    }

    #[test]
    fn sha3_hashing() {
        let _sha3_hasher = sha3::create_sha3_512_hasher(0);
    }
}
