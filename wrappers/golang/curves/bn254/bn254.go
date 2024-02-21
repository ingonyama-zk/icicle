package bn254

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L${SRCDIR}/../../../../icicle/build -lingo_bn254 -lstdc++ -lm
import "C"
