package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/template"
)

const (
	baseDir = "../../curves/" // wrappers/golang/curves
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

type CurveData struct {
	PackageName    string
	Curve          string
	GnarkImport    string
	ScalarLimbsNum int
	BaseLimbsNum   int
	G2BaseLimbsNum int
}

var bn254 = CurveData{
	PackageName:    "bn254",
	Curve:          "bn254",
	GnarkImport:    "bn254",
	ScalarLimbsNum: 4,
	BaseLimbsNum:   4,
	G2BaseLimbsNum: 8,
}
var bls12381 = CurveData{
	PackageName:    "bls12381",
	Curve:          "bls12_381",
	GnarkImport:    "bls12-381",
	ScalarLimbsNum: 4,
	BaseLimbsNum:   6,
	G2BaseLimbsNum: 12,
}
var bls12377 = CurveData{
	PackageName:    "bls12377",
	Curve:          "bls12_377",
	GnarkImport:    "bls12-377",
	ScalarLimbsNum: 4,
	BaseLimbsNum:   6,
	G2BaseLimbsNum: 12,
}
var bw6761 = CurveData{
	PackageName:    "bw6761",
	Curve:          "bw6_761",
	GnarkImport:    "bw6-761",
	ScalarLimbsNum: 6,
	BaseLimbsNum:   12,
	G2BaseLimbsNum: 12,
}

type Entry struct {
	outputName     string
	parsedTemplate *template.Template
}

func toPackage(s string) string {
	return strings.ReplaceAll(s, "-", "")
}

func generateFiles() {
	fmt.Println("Generating files")
	funcs := template.FuncMap{
		"log":       fmt.Println,
		"toLower":   strings.ToLower,
		"toUpper":   strings.ToUpper,
		"toPackage": toPackage,
	}

	curvesData := []CurveData{bn254, bls12377, bls12381, bw6761}
	var entries []Entry

	templateFiles := []string{
		"main.go.tmpl",
		"msm.go.tmpl",
		"msm_test.go.tmpl",
		"ntt.go.tmpl",
		"ntt_test.go.tmpl",
		"curve_test.go.tmpl",
		"curve.go.tmpl",
		"vec_ops_test.go.tmpl",
		"vec_ops.go.tmpl",
		"helpers_test.go.tmpl",
	}

	for _, tmplName := range templateFiles {
		tmpl := template.New(tmplName).Funcs(funcs)
		tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName))
		if err != nil {
			panic(err)
		}
		fileName, ok := strings.CutSuffix(tmplName, ".tmpl")
		if !ok {
			panic(err)
		}
		entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
	}

	templateG2Files := []string{
		"msm.go.tmpl",
		"msm_test.go.tmpl",
		"curve_test.go.tmpl",
		"curve.go.tmpl",
	}

	for _, tmplName := range templateG2Files {
		tmpl := template.New(tmplName).Funcs(funcs)
		tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName))
		if err != nil {
			panic(err)
		}
		rawFileName, _ := strings.CutSuffix(tmplName, ".tmpl")
		fileName := fmt.Sprintf("g2_%s", rawFileName)
		entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
	}

	templateFieldFiles := []string{
		"field.go.tmpl",
		"field_test.go.tmpl",
	}

	fieldFilePrefixes := []string{
		"base",
		"g2_base",
		"scalar",
	}

	for _, tmplName := range templateFieldFiles {
		tmpl := template.New(tmplName).Funcs(funcs)
		tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName), "templates/scalar_field.go.tmpl", "templates/scalar_field_test.go.tmpl")
		if err != nil {
			panic(err)
		}
		fieldFile, _ := strings.CutSuffix(tmplName, ".tmpl")
		for _, fieldPrefix := range fieldFilePrefixes {
			fileName := strings.Join([]string{fieldPrefix, fieldFile}, "_")
			entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
		}
	}

	// Header files
	templateIncludeFiles := []string{
		"curve.h.tmpl",
		"g2_curve.h.tmpl",
		"scalar_field.h.tmpl",
		"msm.h.tmpl",
		"g2_msm.h.tmpl",
		"ntt.h.tmpl",
		"vec_ops.h.tmpl",
	}

	for _, includeFile := range templateIncludeFiles {
		tmpl := template.New(includeFile).Funcs(funcs)
		tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/include/%s", includeFile))
		if err != nil {
			panic(err)
		}
		fileName, _ := strings.CutSuffix(includeFile, ".tmpl")
		entries = append(entries, Entry{outputName: fmt.Sprintf("include/%s", fileName), parsedTemplate: tmplParsed})
	}

	for _, curveData := range curvesData {
		for _, entry := range entries {
			fileName := entry.outputName
			if strings.Contains(fileName, "main") {
				fileName = strings.Replace(fileName, "main", curveData.Curve, 1)
			}
			outFile := filepath.Join(baseDir, curveData.PackageName, fileName)
			var buf bytes.Buffer
			data := struct {
				CurveData
				IsScalar bool
				IsG2     bool
				IsMock   bool
			}{
				curveData,
				strings.Contains(fileName, "scalar_field"),
				strings.Contains(fileName, "g2"),
				false,
			}
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}
	}

	// Generate internal mock field and mock curve files for testing core package
	internalTemplateFiles := []string{
		"curve_test.go.tmpl",
		"curve.go.tmpl",
		"field_test.go.tmpl",
		"field.go.tmpl",
		"helpers_test.go.tmpl",
	}

	for _, internalTemplate := range internalTemplateFiles {
		tmpl := template.New(internalTemplate).Funcs(funcs)
		tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", internalTemplate))
		if err != nil {
			panic(err)
		}
		fileName, _ := strings.CutSuffix(internalTemplate, ".tmpl")
		outFile := filepath.Join("../../core/internal/", fileName)

		var buf bytes.Buffer
		data := struct {
			PackageName  string
			BaseLimbsNum int
			IsMock       bool
			IsG2         bool
			IsScalar     bool
		}{
			"internal",
			4,
			true,
			false,
			false,
		}
		tmplParsed.Execute(&buf, data)
		create(outFile, &buf)
	}
}

func main() {
	generateFiles()
}
