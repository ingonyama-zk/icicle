package bn254_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	icicle "github.com/ingonyama-zk/icicle/goicicle/curves/bn254"
	"github.com/stretchr/testify/assert"
)

func TestNttBN254BatchDIT(t *testing.T) {
	count := 1 << 3
	scalars, frScalars := icicle.GenerateScalars(count, false)

	nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	icicle.NttBatchBN254(&nttResult, false, count, icicle.DIT)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	domain.FFT(frScalars, fft.DIT)

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.ToGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestNttBN254BatchDIF(t *testing.T) {
	count := 1 << 3
	scalars, frScalars := icicle.GenerateScalars(count, false)

	nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	icicle.NttBatchBN254(&nttResult, false, count, icicle.DIF)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	domain.FFT(frScalars, fft.DIF)

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.ToGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestNttBN254CompareToGnarkDIF(t *testing.T) {
	count := 1 << 2
	scalars, frScalars := icicle.GenerateScalars(count, false)

	nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	icicle.NttBN254(&nttResult, false, icicle.DIF)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	// DIT WITH NO INVERSE
	// DIF WITH INVERSE
	domain.FFT(frScalars, fft.DIF) //DIF

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.ToGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestNttBN254CompareToGnarkDIT(t *testing.T) {
	count := 1 << 2
	scalars, frScalars := icicle.GenerateScalars(count, false)

	nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	// not inverse
	icicle.NttBN254(&nttResult, false, icicle.DIT)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	domain.FFT(frScalars, fft.DIT) //DIF

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.ToGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestINttBN254CompareToGnarkDIT(t *testing.T) {
	count := 1 << 3
	scalars, frScalars := icicle.GenerateScalars(count, false)

	nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	icicle.NttBN254(&nttResult, true, icicle.DIT)
	assert.NotEqual(t, nttResult, scalars)

	frResScalars := make([]fr.Element, len(frScalars)) // Make a new slice with the same length
	copy(frResScalars, frScalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	domain.FFTInverse(frResScalars, fft.DIT)

	assert.NotEqual(t, frResScalars, frScalars)

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.ToGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frResScalars)
}

func TestINttBN254CompareToGnarkDIF(t *testing.T) {
	count := 1 << 3
	scalars, frScalars := icicle.GenerateScalars(count, false)

	nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	icicle.NttBN254(&nttResult, true, icicle.DIF)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	domain.FFTInverse(frScalars, fft.DIF)

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.ToGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestNttBN254(t *testing.T) {
	count := 1 << 3

	scalars, _ := icicle.GenerateScalars(count, false)

	nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	icicle.NttBN254(&nttResult, false, icicle.NONE)
	assert.NotEqual(t, nttResult, scalars)

	inttResult := make([]icicle.ScalarField, len(nttResult))
	copy(inttResult, nttResult)

	assert.Equal(t, inttResult, nttResult)
	icicle.NttBN254(&inttResult, true, icicle.NONE)
	assert.Equal(t, inttResult, scalars)
}

func TestNttBatchBN254(t *testing.T) {
	count := 1 << 5
	batches := 4

	scalars, _ := icicle.GenerateScalars(count*batches, false)

	var scalarVecOfVec [][]icicle.ScalarField = make([][]icicle.ScalarField, 0)

	for i := 0; i < batches; i++ {
		start := i * count
		end := (i + 1) * count
		batch := make([]icicle.ScalarField, len(scalars[start:end]))
		copy(batch, scalars[start:end])
		scalarVecOfVec = append(scalarVecOfVec, batch)
	}

	nttBatchResult := make([]icicle.ScalarField, len(scalars))
	copy(nttBatchResult, scalars)

	icicle.NttBatchBN254(&nttBatchResult, false, count, 0)

	var nttResultVecOfVec [][]icicle.ScalarField

	for i := 0; i < batches; i++ {
		// Clone the slice
		clone := make([]icicle.ScalarField, len(scalarVecOfVec[i]))
		copy(clone, scalarVecOfVec[i])

		// Add it to the result vector of vectors
		nttResultVecOfVec = append(nttResultVecOfVec, clone)

		// Call the ntt_bn254 function
		icicle.NttBN254(&nttResultVecOfVec[i], false, icicle.NONE)
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
			scalars, _ := icicle.GenerateScalars(nttSize, false)

			nttResult := make([]icicle.ScalarField, len(scalars)) // Make a new slice with the same length
			copy(nttResult, scalars)
			for n := 0; n < b.N; n++ {
				icicle.NttBN254(&nttResult, false, icicle.NONE)
			}
		})
	}
}
