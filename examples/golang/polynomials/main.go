package main

import (
	"flag"
	"fmt"

	bn254Fft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	bn254Ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ntt"
	bn254Polynomial "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/polynomial"
	runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear"
	babybearNtt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/ntt"
	babybearPolynomial "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/polynomial"
)

var maxNttLogSize uint
var polyLogSize uint

func initBn254Domain() runtime.EIcicleError {
	cfgInitDomain := core.GetDefaultNTTInitDomainConfig()
	rouMontBn254, _ := bn254Fft.Generator(uint64(1 << maxNttLogSize))
	rouBn254 := rouMontBn254.Bits()
	rouIcicleBn254 := bn254.ScalarField{}
	limbsBn254 := core.ConvertUint64ArrToUint32Arr(rouBn254[:])
	rouIcicleBn254.FromLimbs(limbsBn254)
	return bn254Ntt.InitDomain(rouIcicleBn254, cfgInitDomain)
}

func initBabybearDomain() runtime.EIcicleError {
	cfgInitDomain := core.GetDefaultNTTInitDomainConfig()
	rouIcicle := babybear.ScalarField{}
	rouIcicle.FromUint32(1461624142)
	return babybearNtt.InitDomain(rouIcicle, cfgInitDomain)
}

func init() {
	var deviceType string

	flag.UintVar(&maxNttLogSize, "maxNttLogSize", 20, "")
	flag.UintVar(&polyLogSize, "polyLogSize", 15, "")
	flag.StringVar(&deviceType, "device", "CUDA", "Device type")
	flag.Parse()
	if deviceType != "CPU" {
		runtime.LoadBackendFromEnvOrDefault()
	}

	device := runtime.CreateDevice(deviceType, 0)
	// NOTE: If you are only using a single device the entire time
	// 			then this is ok. If you are using multiple devices
	// 			then you should use runtime.RunOnDevice() instead.
	runtime.SetDefaultDevice(&device)

	e := initBn254Domain()
	if e != runtime.Success {
		errorString := fmt.Sprint(
			"Bn254 Domain initialization failed: ", e)
		panic(errorString)
	}
	e = initBabybearDomain()
	if e != runtime.Success {
		errorString := fmt.Sprint(
			"Babybear Domain initialization failed: ", e)
		panic(errorString)
	}
}

func main() {
	polySize := 1 << polyLogSize

	// randomize three polynomials over bn254 scalar field
	var fBn254 bn254Polynomial.DensePolynomial
	var gBn254 bn254Polynomial.DensePolynomial
	var hBn254 bn254Polynomial.DensePolynomial
	fBn254.CreateFromCoeffecitients(bn254.GenerateScalars(polySize))
	gBn254.CreateFromCoeffecitients(bn254.GenerateScalars(polySize / 2))
	hBn254.CreateFromROUEvaluations(bn254.GenerateScalars(polySize / 4))

	// randomize two polynomials over babybear field
	var fBabybear babybearPolynomial.DensePolynomial
	var gBabybear babybearPolynomial.DensePolynomial
	fBabybear.CreateFromCoeffecitients(babybear.GenerateScalars(polySize))
	gBabybear.CreateFromCoeffecitients(babybear.GenerateScalars(polySize / 2))

	// Arithmetic
	t0 := fBn254.Add(&gBn254)
	t1 := fBn254.Multiply(&hBn254)
	q, r := t1.Divide(&t0)
	rBabybear := fBabybear.Add(&gBabybear)
	rDegree := r.Degree()
	_ = rBabybear
	_ = rDegree

	// evaluate in single domain point
	var five bn254.ScalarField
	five.FromUint32(5)
	qAtFive := q.Eval(five)

	var thirty bn254.ScalarField
	thirty.FromUint32(30)

	// evaluate on domain. Note: domain and image can be either Host or Device slice.
	// in this example domain in on host and evals on device.
	hostDomain := core.HostSliceFromElements([]bn254.ScalarField{five, thirty})
	var deviceImage core.DeviceSlice
	_, err := deviceImage.Malloc(five.Size(), hostDomain.Len())
	if err != runtime.Success {
		errorString := fmt.Sprint(
			"deviceImage allocation failed: ", err)
		panic(errorString)
	}
	t1.EvalOnDomain(hostDomain, deviceImage)

	// slicing
	o := hBn254.Odd()
	e := hBn254.Even()

	oddMult := o.MultiplyByScalar(qAtFive)
	fold := e.Add(&oddMult) // e(x) + o(x)*scalar

	coeff := fold.GetCoeff(2) // coeff of x^2
	_ = coeff
}
