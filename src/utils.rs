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

#[cfg(test)]
mod tests {
    use ark_ff::BigInteger256;

    use crate::field::{LimbsField, tests::from_ark_transmute};

    use super::*;

    #[test]
    fn test_u32_vec_to_u64_vec() {
        let arr_u32 = [1, 0x0fffffff, 3, 0x2fffffff, 5, 0x4fffffff, 7, 0x6fffffff];

        let s = from_ark_transmute(BigInteger256::new(
            u32_vec_to_u64_vec(&arr_u32).try_into().unwrap(),
        ))
        .limbs();

        assert_eq!(arr_u32.to_vec(), s);

        let arr_u64_expected = [
            0x0FFFFFFF00000001,
            0x2FFFFFFF00000003,
            0x4FFFFFFF00000005,
            0x6FFFFFFF00000007,
        ];

        assert_eq!(
            u32_vec_to_u64_vec(&arr_u32),
            arr_u64_expected,
            "{:016X?}",
            u32_vec_to_u64_vec(&arr_u32)
        );
    }

    #[test]
    fn test_u64_vec_to_u32_vec() {
        let arr_u64 = [
            0x2FFFFFFF00000001,
            0x4FFFFFFF00000003,
            0x6FFFFFFF00000005,
            0x8FFFFFFF00000007,
        ];

        let arr_u32_expected = [1, 0x2fffffff, 3, 0x4fffffff, 5, 0x6fffffff, 7, 0x8fffffff];

        assert_eq!(
            u64_vec_to_u32_vec(&arr_u64),
            arr_u32_expected,
            "{:016X?}",
            u64_vec_to_u32_vec(&arr_u64)
        );
    }
}
