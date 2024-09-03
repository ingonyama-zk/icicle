package fields

import (
	"path"
	"strings"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

var fieldTemplates = map[string]string{
	"src":    "fields/templates/field.go.tmpl",
	"test":   "fields/templates/field_test.go.tmpl",
	"header": "fields/templates/scalar_field.h.tmpl",
}

func Generate(baseDir, packageName, field, fieldPrefix string, isScalar bool, numLimbs int) {
	data := struct {
		PackageName    string
		Field          string
		FieldPrefix    string
		BaseImportPath string
		IsScalar       bool
		NUM_LIMBS      int
	}{
		packageName,
		field,
		fieldPrefix,
		baseDir,
		isScalar,
		numLimbs,
	}

	filePrefix := ""
	if packageName == "g2" {
		filePrefix = "g2_"
	}

	testDir := "tests"
	parentDir := path.Base(baseDir)
	if parentDir == "g2" || parentDir == "extension" {
		testDir = "../tests"
	}

	generator.GenerateFile(fieldTemplates["src"], baseDir, strings.ToLower(fieldPrefix)+"_", "", data)
	generator.GenerateFile(fieldTemplates["header"], path.Join(baseDir, "include"), "", "", data)
	generator.GenerateFile(fieldTemplates["test"], path.Join(baseDir, testDir), filePrefix+strings.ToLower(fieldPrefix)+"_", "", data)
}
