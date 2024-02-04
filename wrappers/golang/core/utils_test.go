package core

import (
	"testing"
	"github.com/stretchr/testify/assert"
)

func TestConvertUint32ArrToUint64Arr(t *testing.T) {
	testCases := []struct {
		name  string
		input []uint32
		expected  []uint64
	}{
		{
			name:  "Test with incremental array",
			input: []uint32{1, 2, 3, 4, 5, 6, 7, 8},
			expected:  []uint64{4294967298, 12884901892, 21474836486, 30064771080},
		},
		{
			name:  "Test with all zeros",
			input: []uint32{0, 0, 0, 0, 0, 0, 0, 0},
			expected:  []uint64{0, 0, 0, 0},
		},
		{
			name:  "Test with maximum uint32 values",
			input: []uint32{4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295},
			expected:  []uint64{18446744073709551615, 18446744073709551615, 18446744073709551615, 18446744073709551615},
		},
		{
			name:  "Test with alternating min and max uint32 values",
			input: []uint32{0, 4294967295, 0, 4294967295, 0, 4294967295, 0, 4294967295},
			expected:  []uint64{4294967295, 4294967295, 4294967295, 4294967295},
		},
		{
			name:  "Test with alternating max and min uint32 values",
			input: []uint32{4294967295, 0, 4294967295, 0, 4294967295, 0, 4294967295, 0},
			expected:  []uint64{18446744069414584320, 18446744069414584320, 18446744069414584320, 18446744069414584320},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ConvertUint32ArrToUint64Arr(tc.input)
			assert.Equal(t, tc.expected, got, "Got %v,  %v", got, tc.expected)
		})
	}
}

func TestConvertUint64ArrToUint32Arr(t *testing.T) {
	testCases := []struct {
		name     string
		input    []uint64
		expected []uint32
	}{
		{
			name:     "test one",
			input:    []uint64{1, 2, 3, 4},
			expected: []uint32{1, 0, 2, 0, 3, 0, 4, 0},
		},
		{
			name:     "test two",
			input:    []uint64{100, 200, 300, 400},
			expected: []uint32{100, 0, 200, 0, 300, 0, 400, 0},
		},
		{
			name:     "test three",
			input:    []uint64{1000, 2000, 3000, 4000},
			expected: []uint32{1000, 0, 2000, 0, 3000, 0, 4000, 0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ConvertUint64ArrToUint32Arr(tc.input)
			assert.Equal(t, tc.expected, got, "Got %v,  %v", got, tc.expected)
		})
	}
}
