package tests

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

func Generate(baseDir, field, fieldPrefix, gnarkImport string, rou int, supportsNTT bool) {
	data := struct {
		Field          string
		FieldPrefix    string
		BaseImportPath string
		GnarkImport    string
		ROU            int
		SupportsNTT    bool
	}{
		field,
		fieldPrefix,
		baseDir,
		gnarkImport,
		rou,
		supportsNTT,
	}

	generator.GenerateFile("tests/templates/main_test.go.tmpl", path.Join(baseDir, "tests"), "", "", data)
}
