package msm

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/generator_utils"
)

var msmTemplates = map[string]string{
	"src": "msm/templates/msm.go.tmpl",
	"test": "msm/templates/msm_test.go.tmpl",
	"header": "msm/templates/msm.h.tmpl",
}

func Generate(baseDir, packageName, curve, curvePrefix, gnarkImport string) {
	data := struct {
		PackageName string
		Curve string
		CurvePrefix string
		BaseImportPath string
		GnarkImport string
	}{
		packageName,
		curve,
		curvePrefix,
		baseDir,
		gnarkImport,
	}

	generator.GenerateFile(msmTemplates["src"], path.Join(baseDir, packageName), "", "", data)
	generator.GenerateFile(msmTemplates["test"], path.Join(baseDir, packageName), "", "", data)
	generator.GenerateFile(msmTemplates["header"], path.Join(baseDir, packageName, "include"), "", "", data)
}
