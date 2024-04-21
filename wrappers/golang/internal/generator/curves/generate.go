package curves

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/generator_utils"
)

var curveTemplates = map[string]string{
	"src":    "curves/templates/curve.go.tmpl",
	"test":   "curves/templates/curve_test.go.tmpl",
	"header": "curves/templates/curve.h.tmpl",
}

func Generate(baseDir, packageName, curve, curvePrefix string) {
	data := struct {
		PackageName string
		Curve       string
		CurvePrefix string
	}{
		packageName,
		curve,
		curvePrefix,
	}

	generator.GenerateFile(curveTemplates["src"], baseDir, "", "", data)
	generator.GenerateFile(curveTemplates["test"], baseDir, "", "", data)
	generator.GenerateFile(curveTemplates["header"], path.Join(baseDir, "include"), "", "", data)
}
