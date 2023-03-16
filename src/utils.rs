pub fn from_limbs<T>(limbs: Vec<u32>, chunk_size: usize, f: fn(&[u32]) -> T) -> Vec<T> {
    let points = limbs
        .chunks(chunk_size)
        .map(|lmbs| f(lmbs))
        .collect::<Vec<T>>();
    points
}
