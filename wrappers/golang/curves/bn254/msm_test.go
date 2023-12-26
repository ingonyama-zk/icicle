package bn254

import (
	"fmt"
	// "reflect"
	"testing"
	// "time"
	"unsafe"

	"local/hello/icicle/wrappers/golang/core"
	// cr "local/hello/icicle/wrappers/golang/cuda_runtime"
	// "github.com/stretchr/testify/assert"
)

func TestMSM(t *testing.T) {
	// for _, v := range []int{4, 8, 16, 32, 64, 128, 256, 1000, 1 << 18} {
	for _, v := range []int{1} {
		count := 1 << v
		fmt.Println(count)
		
		// TODO: accurately reflect the size of Projective and Affine
		// var p Projective
		// out, e := cr.Malloc(uint(reflect.Tye(p.X).Size()))
		// fmt.Println("Finished malloc")
		// fmt.Println(out)
		// assert.Equal(t, e, cr.CudaSuccess, "error should be nil")
		
		//TODO: getting crazy values for points
		// points := GenerateAffinePoints(count)
		// fmt.Println("Finished generating points")
		// fmt.Println(points[0])
		
		scalars := GenerateScalars(count)
		fmt.Println("Finished generating scalars")
		fmt.Println(scalars)
	

		unsafeScalars := unsafe.Pointer(scalars.AsPointer())
		success := ToMontgomeryUnsafe(unsafeScalars, scalars.Len())
		fmt.Println(success)
		successFrom := FromMontgomery[[]core.Field, uint32](&scalars)
		fmt.Println(successFrom)

		fmt.Println(scalars)

		cfg := GetDefaultMSMConfig()
		fmt.Println("Finished generating cfg MSM")
		fmt.Println(cfg)


		// startTime := time.Now()
		// e := Msm(scalars, points, &cfg, out)
		// fmt.Printf("icicle MSM took: %d ms\n", time.Since(startTime).Milliseconds())

		// assert.Equal(t, e, cr.CudaSuccess, "error should be nil")
	}
}
