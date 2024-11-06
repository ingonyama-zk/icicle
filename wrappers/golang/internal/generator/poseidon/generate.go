package poseidon

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

var poseidonTemplates = map[string]string{
	"src":    "poseidon/templates/poseidon.go.tmpl",
	"test":   "poseidon/templates/poseidon_test.go.tmpl",
	"header": "poseidon/templates/poseidon.h.tmpl",
}

func Generate(baseDir, field string) {
	data := struct {
		BaseImportPath string
		Field          string
	}{
		baseDir,
		field,
	}

	generator.GenerateFile(poseidonTemplates["src"], path.Join(baseDir, "poseidon"), "", "", data)
	generator.GenerateFile(poseidonTemplates["header"], path.Join(baseDir, "poseidon", "include"), "", "", data)
	generator.GenerateFile(poseidonTemplates["test"], path.Join(baseDir, "tests"), "", "", data)
}
