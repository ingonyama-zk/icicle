package fields

import (
	generator "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/generator_utils"
	"strings"
)

var mockTemplates = map[string]string{
	"field": "fields/templates/field.go.tmpl",
	"curve": "curves/templates/curve.go.tmpl",
}

func Generate(baseDir, packageName, field, fieldPrefix string, isScalar bool, numLimbs int) {
	fieldData := struct {
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
	generator.GenerateFile(mockTemplates["field"], baseDir, strings.ToLower(field)+"_", "", fieldData)

	curveData := struct {
		PackageName string
		CurvePrefix string
	}{
		packageName,
		field,
	}
	generator.GenerateFile(mockTemplates["curve"], baseDir, strings.ToLower(field)+"_", "", curveData)
}
