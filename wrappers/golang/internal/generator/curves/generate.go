package curves

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v2/wrappers/golang/internal/generator/generator_utils"
)

var curveTemplates = map[string]string{
	"src":    "curves/templates/curve.go.tmpl",
	"test":   "curves/templates/curve_test.go.tmpl",
	"header": "curves/templates/curve.h.tmpl",
}

func Generate(baseDir, packageName, curve, curvePrefix string) {
	data := struct {
		PackageName    string
		Curve          string
		CurvePrefix    string
		BaseImportPath string
	}{
		packageName,
		curve,
		curvePrefix,
		baseDir,
	}

	filePrefix := ""
	if packageName == "g2" {
		filePrefix = "g2_"
	}

	testDir := "tests"
	parentDir := path.Base(baseDir)
	if parentDir == "g2" || parentDir == "extension" {
		testDir = "../tests"
	}

	generator.GenerateFile(curveTemplates["src"], baseDir, "", "", data)
	generator.GenerateFile(curveTemplates["header"], path.Join(baseDir, "include"), "", "", data)
	generator.GenerateFile(curveTemplates["test"], path.Join(baseDir, testDir), filePrefix, "", data)
}
