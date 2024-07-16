package polynomial

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/internal/generator/generator_utils"
)

var polynomialTemplates = map[string]string{
	"src":    "polynomial/templates/polynomial.go.tmpl",
	"test":   "polynomial/templates/polynomial_test.go.tmpl",
	"header": "polynomial/templates/polynomial.h.tmpl",
}

func Generate(baseDir, field, fieldPrefix, gnarkImport string) {
	data := struct {
		Field          string
		FieldPrefix    string
		BaseImportPath string
		GnarkImport    string
	}{
		field,
		fieldPrefix,
		baseDir,
		gnarkImport,
	}

	generator.GenerateFile(polynomialTemplates["src"], path.Join(baseDir, "polynomial"), "", "", data)
	generator.GenerateFile(polynomialTemplates["header"], path.Join(baseDir, "polynomial", "include"), "", "", data)
	generator.GenerateFile(polynomialTemplates["test"], path.Join(baseDir, "tests"), "", "", data)
}
