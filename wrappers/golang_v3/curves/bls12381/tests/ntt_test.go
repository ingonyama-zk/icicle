package tests

import (
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	bls12_381 "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bls12381"
	ntt "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bls12381/ntt"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/runtime"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/test_helpers"
	"github.com/stretchr/testify/assert"
)

func testAgainstGnarkCryptoNtt(t *testing.T, size int, scalars core.HostSlice[bls12_381.ScalarField], output core.HostSlice[bls12_381.ScalarField], order core.Ordering, direction core.NTTDir) {
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
	expected := test_helpers.GenerateLimbOne(int(bls12_381.SCALAR_LIMBS))
	assert.Equal(t, expected, actual.CosetGen[:])

	cosetGenField := bls12_381.ScalarField{}
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
	scalars := bls12_381.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size

			scalarsCopy := core.HostSliceFromElements[bls12_381.ScalarField](scalars[:testSize])
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[bls12_381.ScalarField], testSize)
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
	scalars := bls12_381.GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				runtime.SetDevice(&DEVICE)

				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[bls12_381.ScalarField](scalars[:testSize])

				stream, _ := runtime.CreateStream()

				cfg.Ordering = v
				cfg.IsAsync = true
				cfg.StreamHandle = stream

				var deviceInput core.DeviceSlice
				scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
				var deviceOutput core.DeviceSlice
				deviceOutput.MallocAsync(testSize*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

				// run ntt
				ntt.Ntt(deviceInput, direction, &cfg, deviceOutput)
				output := make(core.HostSlice[bls12_381.ScalarField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				runtime.SynchronizeStream(stream)
				// Compare with gnark-crypto
				testAgainstGnarkCryptoNtt(t, testSize, scalarsCopy, output, v, direction)
			}
		}
	}
}

func TestNttBatch(t *testing.T) {
	cfg := ntt.GetDefaultNttConfig()
	largestTestSize := 12
	largestBatchSize := 100
	scalars := bls12_381.GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{2, 16, largestBatchSize} {
			runtime.SetDevice(&DEVICE)

			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[bls12_381.ScalarField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)
			// run ntt
			output := make(core.HostSlice[bls12_381.ScalarField], totalSize)
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

// func TestNttArbitraryCoset(t *testing.T) {
// 	for _, size := range []int{20} {
// 		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
// 			testSize := 1 << size
// 			scalars := GenerateScalars(testSize)

// 			cfg := ntt.GetDefaultNttConfig()

// 			var scalarsCopy core.HostSlice[ScalarField]
// 			for _, v := range scalars {
// 				var scalar ScalarField
// 				scalarsCopy = append(scalarsCopy, scalar.FromLimbs(v.GetLimbs()))
// 			}

// 			// init domain
// 			rouMont, _ := fft.Generator(1 << 20)
// 			rou := rouMont.Bits()
// 			rouIcicle := ScalarField{}
// 			limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

// 			rouIcicle.FromLimbs(limbs)
// 			InitDomain(rouIcicle, cfg.Ctx)
// 			cfg.Ordering = v

// 			// run ntt
// 			output := make(core.HostSlice[ScalarField], testSize)
// 			Ntt(scalars, core.KForward, &cfg, output)

// 			// Compare with gnark-crypto
// 			domainWithPrecompute := fft.NewDomain(uint64(testSize))
// 			scalarsFr := make([]fr.Element, testSize)
// 			for i, v := range scalarsCopy {
// 				slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
// 				scalarsFr[i] = slice64
// 			}
// 			outputAsFr := make([]fr.Element, testSize)
// 			for i, v := range output {
// 				slice64, _ := fr.LittleEndian.Element((*[fr.Bytes]byte)(v.ToBytesLittleEndian()))
// 				outputAsFr[i] = slice64
// 			}

// 			// DIT + BitReverse == Ordering.kRR
// 			// DIT == Ordering.kRN
// 			// DIF + BitReverse == Ordering.kNN
// 			// DIF == Ordering.kNR
// 			var decimation fft.Decimation
// 			if v == core.KRN || v == core.KRR {
// 				decimation = fft.DIT
// 			} else {
// 				decimation = fft.DIF
// 			}
// 			domainWithPrecompute.FFT(scalarsFr, decimation, fft.OnCoset())
// 			if v == core.KNN || v == core.KRR {
// 				fft.BitReverse(scalarsFr)
// 			}
// 			if !assert.True(t, reflect.DeepEqual(scalarsFr, outputAsFr)) {
// 				t.FailNow()
// 			}
// 		}
// 	}
// }
