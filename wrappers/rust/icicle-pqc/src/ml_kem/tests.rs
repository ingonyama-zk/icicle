#[cfg(test)]
mod tests {
    use crate::ml_kem::{
        config::MlKemConfig,
        decapsulate, encapsulate, keygen,
        kyber_params::{Kyber1024Params, Kyber512Params, Kyber768Params, KyberParams, ENTROPY_BYTES, MESSAGE_BYTES},
    };
    use icicle_runtime::{
        memory::{DeviceVec, HostSlice, IntoIcicleSlice, IntoIcicleSliceMut},
        runtime,
        stream::IcicleStream,
        Device,
    };
    use rand::Rng;

    fn test_consistency_host<P: KyberParams>(batch_size: usize) {
        let mut public_keys = vec![0; batch_size * P::PUBLIC_KEY_BYTES];
        let mut secret_keys = vec![0; batch_size * P::SECRET_KEY_BYTES];
        let mut messages = vec![0; batch_size * MESSAGE_BYTES];
        let mut ciphertexts = vec![0; batch_size * P::CIPHERTEXT_BYTES];
        let mut shared_secrets_enc = vec![0; batch_size * P::SHARED_SECRET_BYTES];
        let mut shared_secrets_dec = vec![0; batch_size * P::SHARED_SECRET_BYTES];
        let mut entropy = vec![0; batch_size * ENTROPY_BYTES];

        let mut rng = rand::rng();
        rng.fill(&mut entropy[..]);
        rng.fill(&mut messages[..]);

        let mut config = MlKemConfig::default();
        config.batch_size = batch_size;

        keygen::<P>(
            entropy.into_slice(),
            &config,
            public_keys.into_slice_mut(),
            secret_keys.into_slice_mut(),
        )
        .unwrap();

        encapsulate::<P>(
            messages.into_slice(),
            public_keys.into_slice(),
            &config,
            ciphertexts.into_slice_mut(),
            shared_secrets_enc.into_slice_mut(),
        )
        .unwrap();

        decapsulate::<P>(
            secret_keys.into_slice(),
            ciphertexts.into_slice(),
            &config,
            shared_secrets_dec.into_slice_mut(),
        )
        .unwrap();

        assert_eq!(shared_secrets_enc, shared_secrets_dec);
    }

    fn test_consistency_device_async<P: KyberParams>(batch_size: usize) {
        let device = Device::new("CUDA-PQC", 0);
        runtime::set_device(&device).unwrap();

        let mut stream = IcicleStream::create().unwrap();
        runtime::warmup(&stream).unwrap();

        let mut config = MlKemConfig::default();
        config.batch_size = batch_size;
        config.stream = *stream;
        config.is_async = true;

        let mut entropy = vec![0u8; batch_size * ENTROPY_BYTES];
        let mut messages = vec![0u8; batch_size * MESSAGE_BYTES];
        let mut rng = rand::rng();
        rng.fill(&mut entropy[..]);
        rng.fill(&mut messages[..]);

        let mut entropy_d = DeviceVec::device_malloc(batch_size * ENTROPY_BYTES).unwrap();
        let mut messages_d = DeviceVec::device_malloc(batch_size * MESSAGE_BYTES).unwrap();
        entropy_d
            .copy_from_host_async(entropy.into_slice(), &stream)
            .unwrap();
        messages_d
            .copy_from_host_async(messages.into_slice(), &stream)
            .unwrap();

        let mut public_keys_d = DeviceVec::device_malloc(batch_size * P::PUBLIC_KEY_BYTES).unwrap();
        let mut secret_keys_d = DeviceVec::device_malloc(batch_size * P::SECRET_KEY_BYTES).unwrap();
        let mut ciphertexts_d = DeviceVec::device_malloc(batch_size * P::CIPHERTEXT_BYTES).unwrap();
        let mut shared_secrets_enc_d = DeviceVec::device_malloc(batch_size * P::SHARED_SECRET_BYTES).unwrap();
        let mut shared_secrets_dec_d = DeviceVec::device_malloc(batch_size * P::SHARED_SECRET_BYTES).unwrap();

        keygen::<P>(&entropy_d, &config, &mut public_keys_d, &mut secret_keys_d).unwrap();

        encapsulate::<P>(
            &messages_d,
            &public_keys_d,
            &config,
            &mut ciphertexts_d,
            &mut shared_secrets_enc_d,
        )
        .unwrap();

        decapsulate::<P>(&secret_keys_d, &ciphertexts_d, &config, &mut shared_secrets_dec_d).unwrap();

        // Copy results to host for verification
        let mut shared_secrets_enc = vec![0u8; batch_size * P::SHARED_SECRET_BYTES];
        let mut shared_secrets_dec = vec![0u8; batch_size * P::SHARED_SECRET_BYTES];
        shared_secrets_enc_d
            .copy_to_host_async(shared_secrets_enc.into_slice_mut(), &stream)
            .unwrap();
        shared_secrets_dec_d
            .copy_to_host_async(shared_secrets_dec.into_slice_mut(), &stream)
            .unwrap();

        stream
            .synchronize()
            .unwrap();
        stream
            .destroy()
            .unwrap();

        assert_eq!(shared_secrets_enc, shared_secrets_dec);
    }

    fn test_consistency<P: KyberParams>(batch_size: usize) {
        let device = Device::new("CUDA-PQC", 0);
        runtime::set_device(&device).unwrap();
        test_consistency_host::<P>(batch_size);
        test_consistency_device_async::<P>(batch_size);
    }

    #[test]
    fn test_consistency_all_param_sets() {
        const BATCH_SIZE: usize = 1 << 13;
        test_consistency::<Kyber512Params>(BATCH_SIZE);
        test_consistency::<Kyber768Params>(BATCH_SIZE);
        test_consistency::<Kyber1024Params>(BATCH_SIZE);
    }
}
