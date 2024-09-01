package grumpkin

// #cgo LDFLAGS: -L${SRCDIR}/../../../../build/lib -libcicle_field_grumpkin -libcicle_curve_grumpkin -lstdc++ -Wl,-rpath=${SRCDIR}/../../../../build/lib
import "C"
