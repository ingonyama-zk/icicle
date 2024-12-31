package vecops

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

var vecOpsTemplates = map[string]string{
	"src":    "vecOps/templates/vec_ops.go.tmpl",
	"test":   "vecOps/templates/vec_ops_test.go.tmpl",
	"header": "vecOps/templates/vec_ops.h.tmpl",
}

func Generate(baseDir, secondDir, field, fieldPrefix string, secondField string) {
	data := struct {
		PackageName      string
		Field            string
		FieldPrefix      string
		BaseImportPath   string
		SecondField      string
		SecondImportPath string
	}{
		"vecOps",
		field,
		fieldPrefix,
		baseDir,
		secondField,
		secondDir,
	}

	testDir := "tests"
	filePrefix := ""
	parentDir := path.Base(baseDir)
	if parentDir == "extension" {
		testDir = "../tests"
		filePrefix = "extension_"
	}

	generator.GenerateFile(vecOpsTemplates["src"], path.Join(baseDir, "vecOps"), "", "", data)
	generator.GenerateFile(vecOpsTemplates["header"], path.Join(baseDir, "vecOps", "include"), "", "", data)
	generator.GenerateFile(vecOpsTemplates["test"], path.Join(baseDir, testDir), filePrefix, "", data)
}
