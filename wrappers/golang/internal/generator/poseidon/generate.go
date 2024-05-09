package poseidon

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v2/wrappers/golang/internal/generator/generator_utils"
)

var poseidonTemplates = map[string]string{
	"src":    "poseidon/templates/poseidon.go.tmpl",
	"test":   "poseidon/templates/poseidon_test.go.tmpl",
	"header": "poseidon/templates/poseidon.h.tmpl",
}

func Generate(baseDir, additionalDirPath, field, fieldPrefix string) {

	data := struct {
		PackageName    string
		Field          string
		FieldPrefix    string
		BaseImportPath string
	}{
		"poseidon",
		field,
		fieldPrefix,
		baseDir,
	}

	testPath := poseidonTemplates["test"]

	generator.GenerateFile(poseidonTemplates["src"], path.Join(baseDir, additionalDirPath, "poseidon"), "", "", data)
	generator.GenerateFile(poseidonTemplates["header"], path.Join(baseDir, additionalDirPath, "poseidon", "include"), "", "", data)
	generator.GenerateFile(testPath, path.Join(baseDir, "tests"), "", "", data)
}
