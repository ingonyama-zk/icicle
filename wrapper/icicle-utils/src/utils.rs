use std::{
    fs::File,
    io::{Read, Write},
};

use rand::{rngs::StdRng, RngCore, SeedableRng};

pub(crate) fn get_fixed_limbs<const NUM_LIMBS: usize>(val: &[u32]) -> [u32; NUM_LIMBS] {
    match val.len() {
        n if n < NUM_LIMBS => {
            let mut padded: [u32; NUM_LIMBS] = [0; NUM_LIMBS];
            padded[..val.len()].copy_from_slice(&val);
            padded
        }
        n if n == NUM_LIMBS => val.try_into().unwrap(),
        _ => panic!("slice has to much elements"),
    }
}

pub(crate) fn get_rng(seed: Option<u64>) -> Box<dyn RngCore> {
    let rng: Box<dyn RngCore> = match seed {
        Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
        None => Box::new(rand::thread_rng()),
    };
    rng
}

pub(crate) fn export_limbs(numbers: &[u32], file_path: &str) {
    let mut file = File::create(file_path).expect("failed to create file");
    for &number in numbers {
        file.write_all(&number.to_le_bytes())
            .expect("failed to write number to file");
    }
}

pub(crate) fn import_limbs(file_path: &str) -> Vec<u32> {
    let mut file = File::open(file_path).expect("failed to open file");
    let mut buffer = [0; 4];
    let mut result = Vec::new();
    while let Ok(n) = file.read(&mut buffer) {
        if n == 0 {
            break;
        }
        result.push(u32::from_le_bytes(buffer));
    }
    result
}

pub fn from_limbs<T>(limbs: Vec<u32>, chunk_size: usize, f: fn(&[u32]) -> T) -> Vec<T> {
    let points = limbs
        .chunks(chunk_size)
        .map(|lmbs| f(lmbs))
        .collect::<Vec<T>>();
    points
}

fn hex_str_le_to_u32_vec(hex_str: &str, expected_lenght: usize) -> Vec<u32> {
    let hex_str = hex_str.trim_start_matches("0x");
    let rem = hex_str.len() % (8 * expected_lenght);
    let r = if rem == 0 {
        0
    } else {
        8 * expected_lenght - rem
    };
    let hex_str_padded = format!("{}{}", "0".repeat(r), hex_str);

    let ret = hex::decode(hex_str_padded)
        .unwrap()
        .chunks(4)
        .map(|x| u32::from_be_bytes(x.try_into().unwrap()))
        .collect();
    ret
}

pub(crate) fn hex_str_be_to_u32_vec(hex_str: &str, limbsize: usize) -> Vec<u32> {
    let mut ret: Vec<u32> = hex_str_le_to_u32_vec(hex_str, limbsize);
    ret.reverse();
    ret
}

pub fn csv_to_u32_limbs(csv_path: &str, limbsize: usize) -> Vec<u32> {
    let mut hex_strings = Vec::new();
    let file = File::open(csv_path).unwrap();
    let reader = std::io::BufReader::new(file);
    for line in std::io::BufRead::lines(reader) {
        let row = line.unwrap();
        for cell in row.split(',') {
            hex_strings.push(cell.trim().to_string());
        }
    }
    let u32_numbers: Vec<u32> = hex_strings
        .iter()
        .map(|s| hex_str_be_to_u32_vec(s, limbsize))
        .flatten()
        .collect();

    return u32_numbers;
}

fn reverse_bit_order(n: u32, order: u32) -> u32 {
    fn is_power_of_two(n: u32) -> bool {
        n != 0 && n & (n - 1) == 0
    }
    assert!(is_power_of_two(order));
    let mask = order - 1;
    let binary = format!(
        "{:0width$b}",
        n,
        width = (32 - mask.leading_zeros()) as usize
    );
    let reversed = binary.chars().rev().collect::<String>();
    u32::from_str_radix(&reversed, 2).unwrap()
}

pub fn list_to_reverse_bit_order<T: Copy>(l: &[T]) -> Vec<T> {
    l.iter()
        .enumerate()
        .map(|(i, _)| l[reverse_bit_order(i as u32, l.len() as u32) as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::field::SCALAR_LIMBS;

    use super::*;

    #[test]
    fn test_hex_str_to_u32_vec() {
        let hex_str = "32add98ba660b4dd9fcf0a899fc68273abe5fecc0b16f1240df1e93c8c84d";
        let expected_u32_vec = vec![
            0x00032add, 0x98ba660b, 0x4dd9fcf0, 0xa899fc68, 0x273abe5f, 0xecc0b16f, 0x1240df1e,
            0x93c8c84d,
        ];
        let u32_vec = hex_str_le_to_u32_vec(hex_str, SCALAR_LIMBS);
        assert_eq!(u32_vec, expected_u32_vec);
        let hex_str = "0x12039d72be179e9cf6d56ce4a99bd4e8b8ff59367c57bf66f094d4143f7d4ade";
        let expected_u32_vec = vec![
            0x12039d72, 0xbe179e9c, 0xf6d56ce4, 0xa99bd4e8, 0xb8ff5936, 0x7c57bf66, 0xf094d414,
            0x3f7d4ade,
        ];
        let u32_vec = hex_str_le_to_u32_vec(hex_str, SCALAR_LIMBS);
        assert_eq!(u32_vec, expected_u32_vec);
    }

    #[test]
    fn test_reverse_bit_order() {
        assert_eq!(reverse_bit_order(5, 32), 20);
        assert_eq!(reverse_bit_order(7, 64), 56);
        assert_eq!(reverse_bit_order(177, 512), 282);
    }

    #[test]
    fn test_list_to_reverse_bit_order() {
        assert_eq!(list_to_reverse_bit_order(&[0, 1, 2, 3]), &[0, 2, 1, 3]);
        let ll: Vec<u32> = (0..32).collect();
        assert_eq!(
            list_to_reverse_bit_order(&ll),
            &[
                0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13,
                29, 3, 19, 11, 27, 7, 23, 15, 31
            ]
        );
    }
}
