package ntt

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v2/wrappers/golang/internal/generator/generator_utils"
)

var ecnttTemplates = map[string]string{
	"src":    "ecntt/templates/ecntt.go.tmpl",
	"test":   "ecntt/templates/ecntt_test.go.tmpl",
	"header": "ecntt/templates/ecntt.h.tmpl",
}

func Generate(baseDir, curve, gnarkImport string) {
	data := struct {
		Curve          string
		BaseImportPath string
		GnarkImport    string
	}{
		curve,
		baseDir,
		gnarkImport,
	}

	generator.GenerateFile(ecnttTemplates["src"], path.Join(baseDir, "ecntt"), "", "", data)
	generator.GenerateFile(ecnttTemplates["header"], path.Join(baseDir, "ecntt", "include"), "", "", data)
	generator.GenerateFile(ecnttTemplates["test"], path.Join(baseDir, "tests"), "", "", data)
}
