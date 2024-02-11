package bn254

import (
	"local/hello/icicle/wrappers/golang/core"
	cr "local/hello/icicle/wrappers/golang/cuda_runtime"
	"reflect"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/stretchr/testify/assert"
)

func TestNTTGetDefaultConfig(t *testing.T) {
	actual := GetDefaultNttConfig()
	assert.Equal(t, [SCALAR_LIMBS]uint32{1, 0, 0, 0, 0, 0, 0, 0}, actual.CosetGen)

	cosetGenField := ScalarField{}
	cosetGenField.One()
	assert.ElementsMatch(t, cosetGenField.GetLimbs(), actual.CosetGen)
}

func TestInitDomain(t *testing.T) {
	rouMont, _ := fft.Generator(1 << 17)
	rou := rouMont.Bits()
	rouIcicle := ScalarField{}
	limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

	rouIcicle.FromLimbs(limbs)

	ctx, _ := cr.GetDefaultDeviceContext()
	assert.NotPanics(t, func() { InitDomain(rouIcicle, ctx) })
}

func TestNtt(t *testing.T) {
	for _, size := range []int{20} {
		for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
			testSize := 1 << size
			scalars := GenerateScalars(testSize)

			cfg := GetDefaultNttConfig()

			var scalarsCopy core.HostSlice[ScalarField]
			for _, v := range scalars {
				var scalar ScalarField
				scalarsCopy = append(scalarsCopy, scalar.FromLimbs(v.GetLimbs()))
			}

			// init domain
			rouMont, _ := fft.Generator(1 << 20)
			rou := rouMont.Bits()
			rouIcicle := ScalarField{}
			limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

			rouIcicle.FromLimbs(limbs)
			InitDomain(rouIcicle, cfg.Ctx)
			cfg.Ordering = v

			// run ntt
			output := make(core.HostSlice[ScalarField], testSize)
			Ntt(scalars, core.KForward, &cfg, output)

			// Compare with gnark-crypto
			domainWithPrecompute := fft.NewDomain(uint64(testSize))
			scalarsFr := make([]fr.Element, testSize)
			for i, v := range scalarsCopy {
				slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
				scalarsFr[i] = slice64
			}
			outputAsFr := make([]fr.Element, testSize)
			for i, v := range output {
				slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
				outputAsFr[i] = slice64
			}

			// DIT + BitReverse == Ordering.kRR
			// DIT == Ordering.kRN
			// DIF + BitReverse == Ordering.kNN
			// DIF == Ordering.kNR
			var decimation fft.Decimation
			if v == core.KRN || v == core.KRR {
				decimation = fft.DIT
			} else {
				decimation = fft.DIF
			}
			domainWithPrecompute.FFT(scalarsFr, decimation)
			if v == core.KNN || v == core.KRR {
				fft.BitReverse(scalarsFr)
			}
			if !assert.True(t, reflect.DeepEqual(scalarsFr, outputAsFr)) {
				t.FailNow()
			}
		}
	}
}

func TestNttDeviceAsync(t *testing.T) {
	for _, size := range []int{20} {
		for _, direction := range []core.NTTDir{core.KForward, core.KInverse} {
			for _, v := range [4]core.Ordering{core.KNN, core.KNR, core.KRN, core.KRR} {
				testSize := 1 << size
				scalars := GenerateScalars(testSize)

				cfg := GetDefaultNttConfig()

				var scalarsCopy core.HostSlice[ScalarField]
				for _, v := range scalars {
					var scalar ScalarField
					scalarsCopy = append(scalarsCopy, scalar.FromLimbs(v.GetLimbs()))
				}

				// init domain
				rouMont, _ := fft.Generator(1 << 20)
				rou := rouMont.Bits()
				rouIcicle := ScalarField{}
				limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

				rouIcicle.FromLimbs(limbs)
				InitDomain(rouIcicle, cfg.Ctx)
				
				stream, _ := cr.CreateStream()

				cfg.Ordering = v
				cfg.IsAsync = true
				
				var deviceInput core.DeviceSlice
				scalars.CopyToDeviceAsync(&deviceInput, stream, true)
				var deviceOutput core.DeviceSlice
				deviceOutput.MallocAsync(testSize*scalars.SizeOfElement(), scalars.SizeOfElement(), stream)
				
				// run ntt
				Ntt(deviceInput, direction, &cfg, deviceOutput)
				output := make(core.HostSlice[ScalarField], testSize)
				output.CopyFromDeviceAsync(&deviceOutput, stream)

				cr.SynchronizeStream(&stream)

				// Compare with gnark-crypto
				domainWithPrecompute := fft.NewDomain(uint64(testSize))
				scalarsFr := make([]fr.Element, testSize)
				for i, v := range scalarsCopy {
					slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
					scalarsFr[i] = slice64
				}
				outputAsFr := make([]fr.Element, testSize)
				for i, v := range output {
					slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
					outputAsFr[i] = slice64
				}

				// DIT + BitReverse == Ordering.kRR
				// DIT == Ordering.kRN
				// DIF + BitReverse == Ordering.kNN
				// DIF == Ordering.kNR
				var decimation fft.Decimation
				if v == core.KRN || v == core.KRR {
					decimation = fft.DIT
				} else {
					decimation = fft.DIF
				}
				if direction == core.KForward {
					domainWithPrecompute.FFT(scalarsFr, decimation)
					} else {
					domainWithPrecompute.FFTInverse(scalarsFr, decimation)
				}
				if v == core.KNN || v == core.KRR {
					fft.BitReverse(scalarsFr)
				}
				if !assert.True(t, reflect.DeepEqual(scalarsFr, outputAsFr)) {
					t.FailNow()
				}
			}
		}
	}
}

func TestNttBatch(t *testing.T) {
	for _, size := range []int{4, 12} {
		for _, batchSize := range []int{1, 16, 100} {
			testSize := 1 << size
			totalSize := testSize*batchSize
			scalars := GenerateScalars(totalSize)

			cfg := GetDefaultNttConfig()

			var scalarsCopy core.HostSlice[ScalarField]
			for _, v := range scalars {
				var scalar ScalarField
				scalarsCopy = append(scalarsCopy, scalar.FromLimbs(v.GetLimbs()))
			}

			// init domain
			rouMont, _ := fft.Generator(1 << 20)
			rou := rouMont.Bits()
			rouIcicle := ScalarField{}
			limbs := core.ConvertUint64ArrToUint32Arr(rou[:])

			rouIcicle.FromLimbs(limbs)
			InitDomain(rouIcicle, cfg.Ctx)
			cfg.Ordering = core.KNN
			cfg.BatchSize = int32(batchSize)
			// run ntt
			output := make(core.HostSlice[ScalarField], totalSize)
			Ntt(scalars, core.KForward, &cfg, output)

			// Compare with gnark-crypto
			domainWithPrecompute := fft.NewDomain(uint64(testSize))
			outputAsFr := make([]fr.Element, totalSize)
			for i, v := range output {
				slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
				outputAsFr[i] = slice64
			}
			
			for i := 0; i < batchSize; i++ {
				scalarsFr := make([]fr.Element, testSize)
				for i, v := range scalarsCopy[i*testSize:(i+1)*testSize] {
					slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
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
// 				slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
// 				scalarsFr[i] = slice64
// 			}
// 			outputAsFr := make([]fr.Element, testSize)
// 			for i, v := range output {
// 				slice64, _ := fr.LittleEndian.Element((*[32]byte)(v.ToBytesLittleEndian()))
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