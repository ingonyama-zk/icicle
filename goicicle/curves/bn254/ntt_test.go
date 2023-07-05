package bn254

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/stretchr/testify/assert"
)

func TestNttBN254BBB(t *testing.T) {
	count := 1 << 20
	scalars, frScalars := GenerateScalars(count)

	nttResult := make([]ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	NttBatchBN254(&nttResult, false, count, 0)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	// DIT WITH NO INVERSE
	// DIF WITH INVERSE
	domain.FFT(frScalars, fft.DIT) //DIF

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.toGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestNttBN254CompareToGnarkDIF(t *testing.T) {
	count := 1 << 2
	scalars, frScalars := GenerateScalars(count)

	nttResult := make([]ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	NttBN254(&nttResult, false, DIF, 0)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	// DIT WITH NO INVERSE
	// DIF WITH INVERSE
	domain.FFT(frScalars, fft.DIF) //DIF

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.toGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestNttBN254CompareToGnarkDIT(t *testing.T) {
	count := 1 << 2
	scalars, frScalars := GenerateScalars(count)

	nttResult := make([]ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	NttBN254(&nttResult, false, DIT, 0)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	// DIT WITH NO INVERSE
	// DIF WITH INVERSE
	domain.FFT(frScalars, fft.DIT) //DIF

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.toGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestINttBN254CompareToGnarkDIT(t *testing.T) {
	count := 1 << 3
	scalars, frScalars := GenerateScalars(count)

	nttResult := make([]ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	NttBN254(&nttResult, true, DIT, 0)
	assert.NotEqual(t, nttResult, scalars)

	frResScalars := make([]fr.Element, len(frScalars)) // Make a new slice with the same length
	copy(frResScalars, frScalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	domain.FFTInverse(frResScalars, fft.DIT)

	assert.NotEqual(t, frResScalars, frScalars)

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.toGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frResScalars)
}

func TestINttBN254CompareToGnarkDIF(t *testing.T) {
	count := 1 << 3
	scalars, frScalars := GenerateScalars(count)

	nttResult := make([]ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	NttBN254(&nttResult, true, DIF, 0)
	assert.NotEqual(t, nttResult, scalars)

	domain := fft.NewDomain(uint64(len(scalars)))
	domain.FFTInverse(frScalars, fft.DIF)

	nttResultTransformedToGnark := make([]fr.Element, len(scalars)) // Make a new slice with the same length

	for k, v := range nttResult {
		nttResultTransformedToGnark[k] = *v.toGnarkFr()
	}

	assert.Equal(t, nttResultTransformedToGnark, frScalars)
}

func TestNttBN254(t *testing.T) {
	count := 1 << 3

	scalars, _ := GenerateScalars(count)

	nttResult := make([]ScalarField, len(scalars)) // Make a new slice with the same length
	copy(nttResult, scalars)

	assert.Equal(t, nttResult, scalars)
	NttBN254(&nttResult, false, NONE, 0)
	assert.NotEqual(t, nttResult, scalars)

	inttResult := make([]ScalarField, len(nttResult))
	copy(inttResult, nttResult)

	assert.Equal(t, inttResult, nttResult)
	NttBN254(&inttResult, true, NONE, 0)
	assert.Equal(t, inttResult, scalars)
}

func TestNttBatchBN254(t *testing.T) {
	count := 1 << 5
	batches := 4

	scalars, _ := GenerateScalars(count * batches)

	var scalarVecOfVec [][]ScalarField = make([][]ScalarField, 0)

	for i := 0; i < batches; i++ {
		start := i * count
		end := (i + 1) * count
		batch := make([]ScalarField, len(scalars[start:end]))
		copy(batch, scalars[start:end])
		scalarVecOfVec = append(scalarVecOfVec, batch)
	}

	nttBatchResult := make([]ScalarField, len(scalars))
	copy(nttBatchResult, scalars)

	NttBatchBN254(&nttBatchResult, false, count, 0)

	var nttResultVecOfVec [][]ScalarField

	for i := 0; i < batches; i++ {
		// Clone the slice
		clone := make([]ScalarField, len(scalarVecOfVec[i]))
		copy(clone, scalarVecOfVec[i])

		// Add it to the result vector of vectors
		nttResultVecOfVec = append(nttResultVecOfVec, clone)

		// Call the ntt_bn254 function
		NttBN254(&nttResultVecOfVec[i], false, NONE, 0)
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

			nttResult := make([]ScalarField, len(scalars)) // Make a new slice with the same length
			copy(nttResult, scalars)
			for n := 0; n < b.N; n++ {
				NttBN254(&nttResult, false, NONE, 0)
			}
		})
	}
}
