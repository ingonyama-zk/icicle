package tests

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/generator_utils"
)

func Generate(baseDir, field, fieldPrefix, gnarkImport string, rou int) {
	data := struct {
		Field          string
		FieldPrefix    string
		BaseImportPath string
		GnarkImport    string
		ROU            int
	}{
		field,
		fieldPrefix,
		baseDir,
		gnarkImport,
		rou,
	}

	generator.GenerateFile("tests/templates/main_test.go.tmpl", path.Join(baseDir, "tests"), "", "", data)
}
