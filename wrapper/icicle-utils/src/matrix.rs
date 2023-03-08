pub fn split_vec_to_matrix<T: Clone>(vec: &[T], lenght: usize) -> Vec<Vec<T>> {
    vec.chunks(lenght).map(|chunk| chunk.to_vec()).collect()
}

//resize n x m matrix to n*m/k x k
pub fn resize_matrix<T: Clone>(matrix: &Vec<Vec<T>>, k: usize) -> Vec<Vec<T>> {
    let n = matrix.len();
    let m = matrix[0].len();
    let l = m / k;
    let mut result: Vec<Vec<T>> = Vec::with_capacity(n * l);

    for i in 0..n {
        result.append(&mut split_vec_to_matrix(&matrix[i], k));
    }

    result
}

pub fn rows_to_cols<T: Copy>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    (0..matrix[0].len())
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}

pub fn concat_matrices_each<T: Clone>(m1: &[Vec<T>], m2: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut result: Vec<Vec<T>> = Vec::with_capacity(m1.len());

    m1.iter().zip(m2.iter()).for_each(|(row1, row2)| {
        let mut row: Vec<T> = Vec::with_capacity(2 * row1.len());

        row.extend_from_slice(row1);
        row.extend_from_slice(row2);

        result.push(row);
    });

    result
}

pub fn shift_left_and_fill_zeroes<T: Copy + Default>(arr: &mut [T], count: usize) {
    // Shift the elements to the left
    let len = arr.len();
    arr.copy_within(count..len, 0);

    // Fill the second half with zeroes
    let start = len - count;
    let end = len;
    arr[start..end].fill(T::default());
}

pub fn interleave_cols<T: Copy>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    for i in 0..a.len() {
        let mut row = Vec::new();
        for j in 0..a[i].len() {
            row.push(a[i][j]);
            row.push(b[i][j]);
        }
        result.push(row);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rows_to_cols() {
        let matrix: Vec<Vec<u32>> = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25],
        ];

        let all_rows: &[Vec<u32>] = &matrix[..];
        let all_cols = rows_to_cols(&matrix);

        let expected_rows: &[Vec<u32>] = &[
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25],
        ];
        let expected_cols: &[Vec<u32>] = &[
            vec![1, 6, 11, 16, 21],
            vec![2, 7, 12, 17, 22],
            vec![3, 8, 13, 18, 23],
            vec![4, 9, 14, 19, 24],
            vec![5, 10, 15, 20, 25],
        ];

        assert_eq!(all_rows, expected_rows);
        assert_eq!(all_cols, expected_cols);
        assert_eq!(expected_cols[4][2], matrix[2][4])
    }

    #[test]
    fn test_concat_rows_in_matrices() {
        let m1: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let m2: Vec<Vec<i32>> = vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]];
        let expected: Vec<Vec<i32>> = vec![
            vec![1, 2, 3, 10, 20, 30],
            vec![4, 5, 6, 40, 50, 60],
            vec![7, 8, 9, 70, 80, 90],
        ];

        let result = concat_matrices_each(&m1, &m2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_split_vec_to_matrix() {
        let vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let matrix: Vec<Vec<i32>> = vec![vec![1, 2, 3, 4, 5, 6], vec![7, 8, 9, 10, 11, 12]];

        assert_eq!(split_vec_to_matrix(&vec, 6), matrix);
    }

    #[test]
    fn test_split_matrix() {
        let matrix: Vec<Vec<i32>> = vec![vec![1, 2, 3, 4, 5, 6], vec![7, 8, 9, 10, 11, 12]];
        let expected2: Vec<Vec<i32>> = vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![9, 10],
            vec![11, 12],
        ];

        let expected3: Vec<Vec<i32>> = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];

        let result = resize_matrix(&matrix, 2);
        assert_eq!(result, expected2);

        let result = resize_matrix(&matrix, 3);
        assert_eq!(result, expected3);
    }

    #[test]
    fn test_shift_left_and_fill_zeroes() {
        // Test with an even-length array
        let mut arr = [1, 2, 3, 4];
        shift_left_and_fill_zeroes(&mut arr, 2);
        assert_eq!(arr, [3, 4, 0, 0]);

        // Test with an odd-length array
        let mut arr = [1, 2, 3, 4, 5];
        shift_left_and_fill_zeroes(&mut arr, 2);
        assert_eq!(arr, [3, 4, 5, 0, 0]);

        // Test with a string array
        let mut arr = ["a", "b", "c", "d", "e", "f", "g"];
        shift_left_and_fill_zeroes(&mut arr, 4);
        assert_eq!(arr, ["e", "f", "g", "", "", "", ""]);
    }

    #[test]
    fn test_interleave_columns() {
        let a = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let b = vec![vec![11, 12, 13], vec![14, 15, 16]];
        let expected = vec![vec![1, 11, 2, 12, 3, 13], vec![4, 14, 5, 15, 6, 16]];
        assert_eq!(interleave_cols(&a, &b), expected);

        let a = vec![
            vec![1, 2, 3, 4, 5, 6],
            vec![7, 8, 9, 10, 11, 12],
            vec![13, 14, 15, 16, 17, 18],
        ];
        let b = vec![
            vec![100, 101, 102, 103, 104, 105],
            vec![106, 107, 108, 109, 110, 111],
            vec![112, 113, 114, 115, 116, 117],
        ];
        let expected = vec![
            vec![1, 100, 2, 101, 3, 102, 4, 103, 5, 104, 6, 105],
            vec![7, 106, 8, 107, 9, 108, 10, 109, 11, 110, 12, 111],
            vec![13, 112, 14, 113, 15, 114, 16, 115, 17, 116, 18, 117],
        ];
        assert_eq!(interleave_cols(&a, &b), expected);
    }
}
