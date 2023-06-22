use rand::RngCore;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub fn from_limbs<T>(limbs: Vec<u32>, chunk_size: usize, f: fn(&[u32]) -> T) -> Vec<T> {
    let points = limbs
        .chunks(chunk_size)
        .map(|lmbs| f(lmbs))
        .collect::<Vec<T>>();
    points
}

pub fn u32_vec_to_u64_vec(arr_u32: &[u32]) -> Vec<u64> {
    let len = (arr_u32.len() / 2) as usize;
    let mut arr_u64 = vec![0u64; len];

    for i in 0..len {
        arr_u64[i] = u64::from(arr_u32[i * 2]) | (u64::from(arr_u32[i * 2 + 1]) << 32);
    }

    arr_u64
}

pub fn u64_vec_to_u32_vec(arr_u64: &[u64]) -> Vec<u32> {
    let len = arr_u64.len() * 2;
    let mut arr_u32 = vec![0u32; len];

    for i in 0..arr_u64.len() {
        arr_u32[i * 2] = arr_u64[i] as u32;
        arr_u32[i * 2 + 1] = (arr_u64[i] >> 32) as u32;
    }

    arr_u32
}

pub fn get_rng(seed: Option<u64>) -> Box<dyn RngCore> { //TOOD: this func is universal
    let rng: Box<dyn RngCore> = match seed {
        Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
        None => Box::new(rand::thread_rng()),
    };
    rng
}