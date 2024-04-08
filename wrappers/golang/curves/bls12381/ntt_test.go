package bls12381

import (
	"reflect"
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"

	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	"github.com/stretchr/testify/assert"
)

const (
	largestTestSize = 17
)

func init() {
	cfg := GetDefaultNttConfig()
	initDomain(largestTestSize, cfg)
}

func initDomain[T any](largestTestSize int, cfg core.NTTConfig[T]) {
	rouMont, _ := fft.Generator(uint64(1 << largestTestSize))
	rou := rouMont.Bits()
	rouIcicle := ScalarField{}
	limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

	rouIcicle.FromLimbs(limbs)
	InitDomain(rouIcicle, cfg.Ctx, false)
}

func testAgainstGnarkCryptoNtt(size int, scalars core.HostSlice[ScalarField], output core.HostSlice[ScalarField], order core.Ordering, direction core.NTTDir) bool {
	domainWithPrecompute := fft.NewDomain(uint64(size))
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
	return reflect.DeepEqual(scalarsFr, outputAsFr)
}

func TestNTTGetDefaultConfig(t *testing.T) {
	actual := GetDefaultNttConfig()
	expected := generateLimbOne(int(SCALAR_LIMBS))
	assert.Equal(t, expected, actual.CosetGen[:])

	cosetGenField := ScalarField{}
	cosetGenField.One()
	assert.ElementsMatch(t, cosetGenField.GetLimbs(), actual.CosetGen)
}

func TestInitDomain(t *testing.T) {
	t.Skip("Skipped because each test requires the domain to be initialized before running. We ensure this using the init() function")
	cfg := GetDefaultNttConfig()
	assert.NotPanics(t, func() { initDomain(largestTestSize, cfg) })
}

func TestNtt(t *testing.T) {
	cfg := GetDefaultNttConfig()
	scalars := GenerateScalars(1 << largestTestSize)

	for _, size := range []int{4, largestTestSize} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			testSize := 1 << size

			scalarsCopy := core.HostSliceFromElements[ScalarField](scalars[:testSize])
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[ScalarField], testSize)
			Ntt(scalarsCopy, core.KForward, &cfg, output)

			// Compare with gnark-crypto
			assert.True(t, testAgainstGnarkCryptoNtt(testSize, scalarsCopy, output, v, core.KForward))
		}
	}
}

func TestECNtt(t *testing.T) {
	cfg := GetDefaultNttConfig()
	points := GenerateProjectivePoints(1 << largestTestSize)

	for _, size := range []int{4, 5, 6, 7, 8} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			testSize := 1 << size

			pointsCopy := core.HostSliceFromElements[Projective](points[:testSize])
			cfg.Ordering = v
			cfg.NttAlgorithm = core.Radix2

			output := make(core.HostSlice[Projective], testSize)
			e := ECNtt(pointsCopy, core.KForward, &cfg, output)
			assert.Equal(t, core.IcicleErrorCode(0), e.IcicleErrorCode, "ECNtt failed")
		}
	}
}

func TestNttDeviceAsync(t *testing.T) {
	cfg := GetDefaultNttConfig()
	scalars := GenerateScalars(1 << largestTestSize)

	for _, size := range []int{1, 10, largestTestSize} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				testSize := 1 << size
				scalarsCopy := core.HostSliceFromElements[ScalarField](scalars[:testSize])

				stream, _ := cr.CreateStream()

				cfg.Ordering = v
				cfg.IsAsync = true
				cfg.Ctx.Stream = &stream

				var deviceInput core.DeviceSlice
				scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
				var deviceOutput core.DeviceSlice
				deviceOutput.MallocAsync(testSize*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

				// run ntt
				Ntt(deviceInput, direction, &cfg, deviceOutput)
				output := make(core.HostSlice[ScalarField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				cr.SynchronizeStream(&stream)

				// Compare with gnark-crypto
				assert.True(t, testAgainstGnarkCryptoNtt(testSize, scalarsCopy, output, v, direction))
			}
		}
	}
}

func TestNttBatch(t *testing.T) {
	cfg := GetDefaultNttConfig()
	largestBatchSize := 100
	scalars := GenerateScalars(1 << largestTestSize * largestBatchSize)

	for _, size := range []int{4, largestTestSize} {
		for _, batchSize := range []int{1, 16, largestBatchSize} {
			testSize := 1 << size
			totalSize := testSize * batchSize

			scalarsCopy := core.HostSliceFromElements[ScalarField](scalars[:totalSize])

			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)
			// run ntt
			output := make(core.HostSlice[ScalarField], totalSize)
			Ntt(scalarsCopy, core.KForward, &cfg, output)

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

// func TestNttArbitraryCoset(t *testing.T) {
// 	for _, size := range []int{20} {
// 		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
// 			testSize := 1 << size
// 			scalars := GenerateScalars(testSize)

// 			cfg := GetDefaultNttConfig()

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
