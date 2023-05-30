package bn254

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNttBN254(t *testing.T) {
	count := 1 << 3

	scalars, _ := GenerateScalars(count)

	nttResult := make([]FieldBN254, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	NttBN254(&nttResult, false, 0)
	assert.NotEqual(t, nttResult, scalars)

	inttResult := make([]FieldBN254, len(nttResult))
	copy(inttResult, nttResult)

	assert.Equal(t, inttResult, nttResult)
	NttBN254(&inttResult, true, 0)
	assert.Equal(t, inttResult, scalars)
}

func TestNttBatchBN254(t *testing.T) {
	count := 1 << 5
	batches := 4

	scalars, _ := GenerateScalars(count * batches)

	var scalarVecOfVec [][]FieldBN254 = make([][]FieldBN254, 0)

	for i := 0; i < batches; i++ {
		start := i * count
		end := (i + 1) * count
		batch := make([]FieldBN254, len(scalars[start:end]))
		copy(batch, scalars[start:end])
		scalarVecOfVec = append(scalarVecOfVec, batch)
	}

	nttBatchResult := make([]FieldBN254, len(scalars))
	copy(nttBatchResult, scalars)

	NttBatchBN254(&nttBatchResult, false, count, 0)

	var nttResultVecOfVec [][]FieldBN254

	for i := 0; i < batches; i++ {
		// Clone the slice
		clone := make([]FieldBN254, len(scalarVecOfVec[i]))
		copy(clone, scalarVecOfVec[i])

		// Add it to the result vector of vectors
		nttResultVecOfVec = append(nttResultVecOfVec, clone)

		// Call the ntt_bn254 function
		NttBN254(&nttResultVecOfVec[i], false, 0)
	}

	assert.NotEqual(t, nttBatchResult, scalars)

	// Check that the ntt of each vec of scalars is equal to the intt of the specific batch
	for i := 0; i < batches; i++ {
		if !reflect.DeepEqual(nttResultVecOfVec[i], nttBatchResult[i*count:((i+1)*count)]) {
			t.Errorf("ntt of vec of scalars not equal to intt of specific batch")
		}
	}
}

func BenchmarkNTT(b *testing.B) {
	LOG_NTT_SIZES := []int{12, 15, 20, 21, 22, 23, 24, 25, 26}

	for _, logNTTSize := range LOG_NTT_SIZES {
		nttSize := 1 << logNTTSize
		b.Run(fmt.Sprintf("NTT %d", logNTTSize), func(b *testing.B) {
			scalars, _ := GenerateScalars(nttSize)

			nttResult := make([]FieldBN254, len(scalars)) // Make a new slice with the same length
			copy(nttResult, scalars)
			for n := 0; n < b.N; n++ {
				NttBN254(&nttResult, false, 0)
			}
		})
	}
}
