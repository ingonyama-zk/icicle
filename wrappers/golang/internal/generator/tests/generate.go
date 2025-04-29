package tests

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

func Generate(baseDir, field, fieldPrefix, gnarkImport string, supportsNTT bool) {
	data := struct {
		Field          string
		FieldPrefix    string
		BaseImportPath string
		GnarkImport    string
		SupportsNTT    bool
	}{
		field,
		fieldPrefix,
		baseDir,
		gnarkImport,
		supportsNTT,
	}

	generator.GenerateFile("tests/templates/main_test.go.tmpl", path.Join(baseDir, "tests"), "", "", data)
}
