package ntt

import (
	"path"

	generator "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/generator_utils"
)

var nttTemplates = map[string]string{
	"src":          "ntt/templates/ntt.go.tmpl",
	"test":         "ntt/templates/ntt_test.go.tmpl",
	"testNoDomain": "ntt/templates/ntt_no_domain_test.go.tmpl",
	"header":       "ntt/templates/ntt.h.tmpl",
}

var ecnttTemplates = map[string]string{
	"src":    "ntt/templates/ecntt.go.tmpl",
	"test":   "ntt/templates/ecntt_test.go.tmpl",
	"header": "ntt/templates/ecntt.h.tmpl",
}

func Generate(baseDir, additionalDirPath, field, fieldPrefix, gnarkImport string, rou int, withDomain bool, fieldNoDomain, fieldNoDomainPrefix string) {
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
		ROU                    int
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
		rou,
		fieldNoDomain,
		fieldNoDomainPrefix,
		baseImportPathNoDomain,
	}

	testPath := nttTemplates["test"]
	if !withDomain {
		testPath = nttTemplates["testNoDomain"]
	}

	generator.GenerateFile(nttTemplates["src"], path.Join(baseDir, additionalDirPath, "ntt"), "", "", data)
	generator.GenerateFile(testPath, path.Join(baseDir, additionalDirPath, "ntt"), "", "", data)
	generator.GenerateFile(nttTemplates["header"], path.Join(baseDir, additionalDirPath, "ntt", "include"), "", "", data)
}
