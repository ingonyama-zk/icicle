package ntt

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

var nttTemplates = map[string]string{
	"src":    "ntt/templates/ntt.go.tmpl",
	"test":   "ntt/templates/ntt_test.go.tmpl",
	"header": "ntt/templates/ntt.h.tmpl",
}

func Generate(baseDir, additionalDirPath, field, fieldPrefix, gnarkImport string, withDomain bool, fieldNoDomain, fieldNoDomainPrefix string) {
	baseImportPathNoDomain := ""
	if !withDomain {
		baseImportPathNoDomain = path.Join(baseDir, additionalDirPath)
	}

	data := struct {
		PackageName            string
		Field                  string
		FieldPrefix            string
		WithDomain             bool
		BaseImportPath         string
		GnarkImport            string
		FieldNoDomain          string
		FieldNoDomainPrefix    string
		BaseImportPathNoDomain string
	}{
		"ntt",
		field,
		fieldPrefix,
		withDomain,
		baseDir,
		gnarkImport,
		fieldNoDomain,
		fieldNoDomainPrefix,
		baseImportPathNoDomain,
	}

	testPath := nttTemplates["test"]

	generator.GenerateFile(nttTemplates["src"], path.Join(baseDir, additionalDirPath, "ntt"), "", "", data)
	generator.GenerateFile(nttTemplates["header"], path.Join(baseDir, additionalDirPath, "ntt", "include"), "", "", data)
	generator.GenerateFile(testPath, path.Join(baseDir, "tests"), "", "", data)
}
