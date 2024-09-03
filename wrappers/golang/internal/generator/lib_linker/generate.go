package lib_linker

import (
	"strings"

	generator "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/generator_utils"
)

type MainTemplateType string

const (
	CURVE MainTemplateType = "curves"
	FIELD MainTemplateType = "fields"
)

var mainTemplates = map[MainTemplateType]string{
	"fields": "fields/templates/main.go.tmpl",
	"curves": "curves/templates/main.go.tmpl",
}

func Generate(baseDir, packageName, field string, templateType MainTemplateType, numAdditionalDirectoriesToLib int) {
	data := struct {
		PackageName string
		Field       string
		UpDirs      string
	}{
		packageName,
		field,
		strings.Repeat("../", numAdditionalDirectoriesToLib),
	}

	generator.GenerateFile(mainTemplates[templateType], baseDir, "", "", data)
}
