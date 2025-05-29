use icicle_runtime::memory::{HostSlice, HostOrDeviceSlice};
use icicle_runtime::{device::Device, is_device_available, get_active_device, set_device, runtime::load_backend_from_env_or_default};
use crate::traits::{GenerateRandom, FieldImpl};
use crate::vec_ops::{VecOps, VecOpsConfig};
use std::{env, sync::OnceLock};

#[macro_export]
macro_rules! impl_matrix_mult_bench {
    (
      $field_prefix:literal,
      $field:ident
    ) => {
        use std::{env, sync::OnceLock};
        use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
        use icicle_runtime::{memory::{HostSlice, HostOrDeviceSlice}, device::Device, is_device_available, get_active_device, set_device, runtime::load_backend_from_env_or_default};
        use icicle_core::{
            traits::{GenerateRandom, FieldImpl},
            vec_ops::{VecOps, matrix_mult, tq_matrix_mult, VecOpsConfig},
        };

        static INIT: OnceLock<()> = OnceLock::new();

        fn load_and_init_backend_device() {
            // Attempt to load the backends
            let _ = load_backend_from_env_or_default(); // try loading from /opt/icicle/backend or env ${ICICLE_BACKEND_INSTALL_DIR}

            // Check if BENCH_TARGET is defined
            let target = env::var("BENCH_TARGET").unwrap_or_else(|_| {
                // If not defined, try CUDA first, fallback to CPU
                if is_device_available(&Device::new("CUDA", 0)) {
                    "CUDA".to_string()
                } else {
                    "CPU".to_string()
                }
            });

            // Initialize the device with the determined target
            let device = Device::new(&target, 0);
            set_device(&device).unwrap();

            println!("ICICLE benchmark with {:?}", device);
        }

        fn benchmark_matrix_mult<F: FieldImpl>(c: &mut Criterion)
        where
            <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F>,
        {
            use criterion::SamplingMode;
            use std::env;

            load_and_init_backend_device();

            let mut group_standard = c.benchmark_group(format!("{} Standard Matrix Multiplication", $field_prefix));
            //let mut group_tq = c.benchmark_group(format!("{} TQ Matrix Multiplication", $field_prefix));
            
            group_standard.sampling_mode(SamplingMode::Flat);
            group_standard.sample_size(10);
            
            //group_tq.sampling_mode(SamplingMode::Flat);
            //group_tq.sample_size(10);

            // Configuration for vector operations
            let cfg = VecOpsConfig::default();

            // Sizes to benchmark (rows x columns)
            let sizes = vec![64, 128, 256, 512];
            
            for size in sizes.iter() {
                let size = *size;
                
                // Standard matrix multiplication benchmark
                group_standard.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
                    // Square matrices of the given size
                    let mat_a = F::Config::generate_random(size * size);
                    let mat_b = F::Config::generate_random(size * size);
                    let mut result = vec![F::zero(); size * size];
                    
                    let mat_a_slice = HostSlice::from_slice(&mat_a);
                    let mat_b_slice = HostSlice::from_slice(&mat_b);
                    let result_slice = HostSlice::from_mut_slice(&mut result);
                    
                    b.iter(|| {
                        matrix_mult(
                            mat_a_slice,
                            black_box(size as u32),
                            black_box(size as u32),
                            mat_b_slice,
                            black_box(size as u32),
                            black_box(size as u32),
                            result_slice,
                            black_box(&cfg),
                        ).unwrap();
                    });
                });
                
                // // TQ matrix multiplication benchmark
                // // We'll try different dimension parameters
                // for d in [1, 2, 4].iter() {
                //     group_tq.bench_with_input(BenchmarkId::new("Size", format!("{size}x{size}_d{d}")), &size, |b, &size| {
                //         // Square matrices of the given size
                //         let mat_a = F::Config::generate_random(size * size * d);
                //         let mat_b = F::Config::generate_random(size * size * d);
                //         let mut result = vec![F::zero(); size * size * d];
                        
                //         let mat_a_slice = HostSlice::from_slice(&mat_a);
                //         let mat_b_slice = HostSlice::from_slice(&mat_b);
                //         let result_slice = HostSlice::from_mut_slice(&mut result);
                        
                //         b.iter(|| {
                //             tq_matrix_mult(
                //                 black_box(*d as u32),
                //                 black_box(mat_a_slice),
                //                 black_box(size as u32),
                //                 black_box(size as u32),
                //                 black_box(mat_b_slice),
                //                 black_box(size as u32),
                //                 black_box(size as u32),
                //                 black_box(result_slice),
                //                 black_box(&cfg),
                //             ).unwrap();
                //         });
                //     });
                // }
            }

            group_standard.finish();
            //group_tq.finish();
        }

        criterion_group!(benches, benchmark_matrix_mult::<$field>);
        criterion_main!(benches);
    };
} 