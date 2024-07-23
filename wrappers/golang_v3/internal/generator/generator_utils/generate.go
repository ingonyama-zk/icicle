package generator_utils

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"text/template"
)

const (
	// Since path.Join joins from the cwd we only need to go up two directories
	// from wrappers/golang_v3/internal/generator/main.go to get to wrappers/golang
	GOLANG_WRAPPER_ROOT_DIR = "../../"
)

func create(output string, buf *bytes.Buffer) error {
	// create output dir if not exist
	_ = os.MkdirAll(filepath.Dir(output), os.ModePerm)

	// create output file
	file, err := os.Create(output)
	if err != nil {
		return err
	}

	if _, err := io.Copy(file, buf); err != nil {
		file.Close()
		return err
	}

	file.Close()
	return nil
}

type entry struct {
	outputName     string
	parsedTemplate *template.Template
}

func toPackage(s string) string {
	return strings.ReplaceAll(s, "-", "")
}

func toCName(s string) string {
	if s == "" {
		return ""
	}
	return strings.ToLower(s) + "_"
}

func toCNameBackwards(s string) string {
	if s == "" {
		return ""
	}
	return "_" + strings.ToLower(s)
}

func toConst(s string) string {
	if s == "" {
		return ""
	}
	return strings.ToUpper(s) + "_"
}
func capitalize(s string) string {
	if s == "" {
		return ""
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

var templateFuncs = template.FuncMap{
	"log":              fmt.Println,
	"toLower":          strings.ToLower,
	"toUpper":          strings.ToUpper,
	"toPackage":        toPackage,
	"toCName":          toCName,
	"toCNameBackwards": toCNameBackwards,
	"toConst":          toConst,
	"capitalize":       capitalize,
}

func parseTemplateFile(tmplPath string) entry {
	tmplName := tmplPath[strings.LastIndex(tmplPath, "/")+1:]
	tmpl := template.New(tmplName).Funcs(templateFuncs)
	tmplParsed, err := tmpl.ParseFiles(tmplPath)
	if err != nil {
		panic(err)
	}
	fileName, ok := strings.CutSuffix(tmplName, ".tmpl")
	if !ok {
		panic(".tmpl suffix not found")
	}

	return entry{outputName: fileName, parsedTemplate: tmplParsed}
}

func GenerateFile(templateFilePath, baseDir, fileNamePrefix, fileNameSuffix string, data any) {
	entry := parseTemplateFile(templateFilePath)
	var buf bytes.Buffer
	entry.parsedTemplate.Execute(&buf, data)
	outFile := path.Join(GOLANG_WRAPPER_ROOT_DIR, baseDir, fileNamePrefix+entry.outputName+fileNameSuffix)
	create(outFile, &buf)
}
