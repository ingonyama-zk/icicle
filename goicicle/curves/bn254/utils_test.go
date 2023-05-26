package bn254

import (
	"testing"
)

func TestConvertUint32ArrToUint64Arr(t *testing.T) {
	testCases := []struct {
		name  string
		input [8]uint32
		want  [4]uint64
	}{
		{
			name:  "Test with incremental array",
			input: [8]uint32{1, 2, 3, 4, 5, 6, 7, 8},
			want:  [4]uint64{4294967298, 12884901892, 21474836486, 30064771080},
		},
		{
			name:  "Test with all zeros",
			input: [8]uint32{0, 0, 0, 0, 0, 0, 0, 0},
			want:  [4]uint64{0, 0, 0, 0},
		},
		{
			name:  "Test with maximum uint32 values",
			input: [8]uint32{4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295},
			want:  [4]uint64{18446744073709551615, 18446744073709551615, 18446744073709551615, 18446744073709551615},
		},
		{
			name:  "Test with alternating min and max uint32 values",
			input: [8]uint32{0, 4294967295, 0, 4294967295, 0, 4294967295, 0, 4294967295},
			want:  [4]uint64{4294967295, 4294967295, 4294967295, 4294967295},
		},
		{
			name:  "Test with alternating max and min uint32 values",
			input: [8]uint32{4294967295, 0, 4294967295, 0, 4294967295, 0, 4294967295, 0},
			want:  [4]uint64{18446744069414584320, 18446744069414584320, 18446744069414584320, 18446744069414584320},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ConvertUint32ArrToUint64Arr(tc.input)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestConvertUint64ArrToUint32Arr(t *testing.T) {
	testCases := []struct {
		name  string
		input [4]uint64
		want  [8]uint32
	}{
		{
			name:  "Test with incremental array",
			input: [4]uint64{4294967298, 12884901892, 21474836486, 30064771080},
			want:  [8]uint32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:  "Test with all zeros",
			input: [4]uint64{0, 0, 0, 0},
			want:  [8]uint32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:  "Test with maximum uint64 values",
			input: [4]uint64{18446744073709551615, 18446744073709551615, 18446744073709551615, 18446744073709551615},
			want:  [8]uint32{4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295},
		},
		{
			name:  "Test with alternating min and max uint64 values",
			input: [4]uint64{4294967295, 4294967295, 4294967295, 4294967295},
			want:  [8]uint32{0, 4294967295, 0, 4294967295, 0, 4294967295, 0, 4294967295},
		},
		{
			name:  "Test with alternating max and min uint64 values",
			input: [4]uint64{18446744069414584320, 18446744069414584320, 18446744069414584320, 18446744069414584320},
			want:  [8]uint32{4294967295, 0, 4294967295, 0, 4294967295, 0, 4294967295, 0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ConvertUint64ArrToUint32Arr(tc.input)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}
