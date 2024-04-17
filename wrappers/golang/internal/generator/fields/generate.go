package fields

import (
	generator "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/generator_utils"
	"path"
	"strings"
)

var fieldTemplates = map[string]string{
	"src":    "fields/templates/field.go.tmpl",
	"test":   "fields/templates/field_test.go.tmpl",
	"header": "fields/templates/scalar_field.h.tmpl",
}

func Generate(baseDir, packageName, field, fieldPrefix string, isScalar bool, numLimbs int) {
	data := struct {
		PackageName string
		Field       string
		FieldPrefix string
		IsScalar    bool
		NUM_LIMBS   int
	}{
		packageName,
		field,
		fieldPrefix,
		isScalar,
		numLimbs,
	}

	generator.GenerateFile(fieldTemplates["src"], baseDir, strings.ToLower(fieldPrefix)+"_", "", data)
	generator.GenerateFile(fieldTemplates["test"], baseDir, strings.ToLower(fieldPrefix)+"_", "", data)
	generator.GenerateFile(fieldTemplates["header"], path.Join(baseDir, "include"), "", "", data)
}
