package msm

import (
	"path"
	"path/filepath"

	generator "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/internal/generator/generator_utils"
)

var msmTemplates = map[string]string{
	"src":    "msm/templates/msm.go.tmpl",
	"test":   "msm/templates/msm_test.go.tmpl",
	"header": "msm/templates/msm.h.tmpl",
}

func Generate(baseDir, packageName, curve, curvePrefix, gnarkImport string) {
	data := struct {
		PackageName    string
		Curve          string
		CurvePrefix    string
		BaseImportPath string
		GnarkImport    string
	}{
		packageName,
		curve,
		curvePrefix,
		baseDir,
		gnarkImport,
	}

	filePrefix := ""
	if packageName == "g2" {
		filePrefix = "g2_"
	}

	generator.GenerateFile(msmTemplates["src"], path.Join(baseDir, packageName), "", "", data)
	generator.GenerateFile(msmTemplates["header"], path.Join(baseDir, packageName, "include"), "", "", data)
	generator.GenerateFile(msmTemplates["test"], filepath.Join(baseDir, "tests"), filePrefix, "", data)
}
