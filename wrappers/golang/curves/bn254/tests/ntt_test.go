package tests

import (
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bn254 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/test_helpers"
	"github.com/stretchr/testify/assert"
)

func testAgainstGnarkCryptoNtt(t *testing.T, size int, scalars core.HostSlice[bn254.ScalarField], output core.HostSlice[bn254.ScalarField], order core.Ordering, direction core.NTTDir) {
	scalarsFr := make([]fr.Element, size)
	for i, v := range scalars {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		scalarsFr[i] = slice64
	}
	outputAsFr := make([]fr.Element, size)
	for i, v := range output {
		slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
		outputAsFr[i] = slice64
	}

	testAgainstGnarkCryptoNttGnarkTypes(t, size, scalarsFr, outputAsFr, order, direction)
}

func testAgainstGnarkCryptoNttGnarkTypes(t *testing.T, size int, scalarsFr core.HostSlice[fr.Element], outputAsFr core.HostSlice[fr.Element], order core.Ordering, direction core.NTTDir) {
	domainWithPrecompute := fft.NewDomain(uint64(size))
	// DIT + BitReverse == Ordering.kRR
	// DIT == Ordering.kRN
	// DIF + BitReverse == Ordering.kNN
	// DIF == Ordering.kNR
	var decimation fft.Decimation
	if order == core.KRN || order == core.KRR {
		decimation = fft.DIT
	} else {
		decimation = fft.DIF
	}

	if direction == core.KForward {
		domainWithPrecompute.FFT(scalarsFr, decimation)
	} else {
		domainWithPrecompute.FFTInverse(scalarsFr, decimation)
	}

	if order == core.KNN || order == core.KRR {
		fft.BitReverse(scalarsFr)
	}
	assert.Equal(t, scalarsFr, outputAsFr)
}
func TestNTTGetDefaultConfig(t *testing.T) {
	actual := ntt.GetDefaultNttConfig()
	expected := test_helpers.GenerateLimbOne(int(bn254.SCALAR_LIMBS))
	assert.Equal(t, expected, actual.CosetGen[:])

	cosetGenField := bn254.ScalarField{}
	cosetGenField.One()
	assert.ElementsMatch(t, cosetGenField.GetLimbs(), actual.CosetGen)
}

func TestInitDomain(t *testing.T) {
	t.Skip("Skipped because each test requires the domain to be initialized before running. We ensure this using the TestMain() function")
	cfg := core.GetDefaultNTTInitDomainConfig()
	assert.NotPanics(t, func() { initDomain(largestTestSize, cfg) })
}

func TestNtt(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := bn254.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size

			scalarsCopy := core.HostSliceFromElements[bn254.ScalarField](scalars[:testSize])
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[bn254.ScalarField], testSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)

			// Compare with gnark-crypto
			testAgainstGnarkCryptoNtt(t, testSize, scalarsCopy, output, v, core.KForward)
		}
	}
}
func TestNttFrElement(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := make([]fr.Element, 4)
	var x fr.Element
	for i := 0; i < 4; i++ {
		x.SetRandom()
		scalars[i] = x
	}

	for _, size := range []int{4} {
		for _, v := range [1]core.Ordering{core.KNN} {
			runtime.SetDevice(&DEVICE)

			testSize := size

			scalarsCopy := (core.HostSlice[fr.Element])(scalars[:testSize])
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[fr.Element], testSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)

			// Compare with gnark-crypto
			testAgainstGnarkCryptoNttGnarkTypes(t, testSize, scalarsCopy, output, v, core.KForward)
		}
	}
}

func TestNttDeviceAsync(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	scalars := bn254.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				runtime.SetDevice(&DEVICE)

				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[bn254.ScalarField](scalars[:testSize])

				stream, _ := runtime.CreateStream()

				cfg.Ordering = v
				cfg.IsAsync = true
				cfg.StreamHandle = stream

				var deviceInput core.DeviceSlice
				scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
				var deviceOutput core.DeviceSlice
				deviceOutput.MallocAsync(scalarsCopy.SizeOfElement(), testSize, stream)

				// run ntt
				ntt.Ntt(deviceInput, direction, &cfg, deviceOutput)
				output := make(core.HostSlice[bn254.ScalarField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				runtime.SynchronizeStream(stream)
				runtime.DestroyStream(stream)
				// Compare with gnark-crypto
				testAgainstGnarkCryptoNtt(t, testSize, scalarsCopy, output, v, direction)
			}
		}
	}
}

func TestNttBatch(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	largestTestSize := 10
	largestBatchSize := 20
	scalars := bn254.GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{2, 16, largestBatchSize} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[bn254.ScalarField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)
			// run ntt
			output := make(core.HostSlice[bn254.ScalarField], totalSize)
			ntt.Ntt(scalarsCopy, core.KForward, &cfg, output)

			// Compare with gnark-crypto
			domainWithPrecompute := fft.NewDomain(uint64(testSize))
			outputAsFr := make([]fr.Element, totalSize)
			for i, v := range output {
				slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
				outputAsFr[i] = slice64
			}

			for i := 0; i < batchSize; i++ {
				scalarsFr := make([]fr.Element, testSize)
				for i, v := range scalarsCopy[i*testSize : (i+1)*testSize] {
					slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
					scalarsFr[i] = slice64
				}

				domainWithPrecompute.FFT(scalarsFr, fft.DIF)
				fft.BitReverse(scalarsFr)
				if !assert.True(t, reflect.DeepEqual(scalarsFr, outputAsFr[i*testSize:(i+1)*testSize])) {
					t.FailNow()
				}
			}
		}
	}
}

func TestReleaseDomain(t *testing.T) {
	t.Skip("Skipped because each test requires the domain to be initialized before running. We ensure this using the TestMain() function")
	e := ntt.ReleaseDomain()
	assert.Equal(t, runtime.Success, e, "ReleasDomain failed")
}
