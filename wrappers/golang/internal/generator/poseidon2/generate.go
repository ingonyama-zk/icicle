package poseidon2

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

var poseidonTemplates = map[string]string{
	"src":    "poseidon2/templates/poseidon2.go.tmpl",
	"test":   "poseidon2/templates/poseidon2_test.go.tmpl",
	"header": "poseidon2/templates/poseidon2.h.tmpl",
}

func Generate(baseDir, field string) {
	data := struct {
		BaseImportPath string
		Field          string
	}{
		baseDir,
		field,
	}

	generator.GenerateFile(poseidonTemplates["src"], path.Join(baseDir, "poseidon2"), "", "", data)
	generator.GenerateFile(poseidonTemplates["header"], path.Join(baseDir, "poseidon2", "include"), "", "", data)
	generator.GenerateFile(poseidonTemplates["test"], path.Join(baseDir, "tests"), "", "", data)
}
