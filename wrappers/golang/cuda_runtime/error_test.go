package cuda_runtime

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetLastError(t *testing.T) {
	assert.NotPanicsf(t, func() { GetLastError() }, "Call to cuda GetLastError panicked")
}
