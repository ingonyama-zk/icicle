package vecops

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/generator_utils"
)

var vecOpsTemplates = map[string]string{
	"src":    "vecOps/templates/vec_ops.go.tmpl",
	"test":   "vecOps/templates/vec_ops_test.go.tmpl",
	"header": "vecOps/templates/vec_ops.h.tmpl",
}

func Generate(baseDir, field, fieldPrefix string) {
	data := struct {
		PackageName    string
		Field          string
		FieldPrefix    string
		BaseImportPath string
	}{
		"vecOps",
		field,
		fieldPrefix,
		baseDir,
	}

	generator.GenerateFile(vecOpsTemplates["src"], path.Join(baseDir, "vecOps"), "", "", data)
	generator.GenerateFile(vecOpsTemplates["test"], path.Join(baseDir, "vecOps"), "", "", data)
	generator.GenerateFile(vecOpsTemplates["header"], path.Join(baseDir, "vecOps", "include"), "", "", data)
}
