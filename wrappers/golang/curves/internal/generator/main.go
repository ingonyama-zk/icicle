package main

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"os"
	"path/filepath"
	"text/template"
)

const (
	baseDir         = "../../" // wrappers/golang/curves
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
	PackageName     string
	Curve        		string
	GnarkImport     string
	ScalarLimbsNum  int
	BaseLimbsNum    int
	G2BaseLimbsNum  int
	IsG2 						bool
	IsScalar 				bool
}

var bn254 = CurveData{
	PackageName:    "bn254",
	Curve:        	"bn254",
	GnarkImport:    "bn254",
	ScalarLimbsNum: 8,
	BaseLimbsNum:   8,
	G2BaseLimbsNum: 16,
	IsG2: 					false,
	IsScalar: 			false,
}
var bls12381 = CurveData{
	PackageName:    "bls12381",
	Curve:        	"bls12_381",
	GnarkImport:    "bls12-381",
	ScalarLimbsNum: 8,
	BaseLimbsNum:   12,
	G2BaseLimbsNum: 24,
	IsG2: 					false,
	IsScalar: 			false,
}
var bls12377 = CurveData{
	PackageName:    "bls12377",
	Curve:        	"bls12_377",
	GnarkImport:    "bls12-377",
	ScalarLimbsNum: 8,
	BaseLimbsNum:   12,
	G2BaseLimbsNum: 24,
	IsG2: 					false,
	IsScalar: 			false,
}
// var bw6761 = CurveData{
// 	PackageName:        	"bw6761",
// 	Curve:        		"bw6_761",
// 	ScalarLimbsNum: 8,
// 	BaseLimbsNum:   12,
// 	G2BaseLimbsNum: 16,
// 	IsG2: false,
// 	IsScalar: false,
// }

type Entry struct {
	outputName string
	parsedTemplate *template.Template
}

func generateFiles() {
	fmt.Println("Generating files")
	funcs := template.FuncMap{
		"log":       fmt.Println,
		"toLower":   strings.ToLower,
		"toUpper":   strings.ToUpper,
	}

	curvesData := []CurveData{bn254, bls12377, bls12381}
	var entries []Entry

	for _, tmplName := range []string{"msm.go.tmpl", "ntt.go.tmpl", "curve.go.tmpl"/*, "vec_ops.h.tmpl"*/} {
		tmpl := template.New(tmplName)
		tmplParsed, _ := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName))
		fileName, _ := strings.CutSuffix(tmplName, ".tmpl")
		entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
	}

	tmplName := "field.go.tmpl"
	tmpl := template.New(tmplName)
	tmplParsed, _ := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName))
	fieldFile, _ := strings.CutSuffix(tmplName, ".tmpl")
	fileName := strings.Join([]string{"base", fieldFile}, "_")
	entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
	
	tmplName = "field.go.tmpl"
	tmpl = template.New(tmplName)
	tmplParsed, _ = tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName), "templates/scalar_field.go.tmpl")
	fieldFile, _ = strings.CutSuffix(tmplName, ".tmpl")
	fileName = strings.Join([]string{"scalar", fieldFile}, "_")
	entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})

	for _, includeFile := range []string{"curve.h.tmpl", "scalar_field.h.tmpl", "msm.h.tmpl", "ntt.h.tmpl"/*, "vec_ops.h.tmpl"*/} {
		tmpl := template.New(includeFile).Funcs(funcs)
		tmplParsed, _ := tmpl.ParseFiles(fmt.Sprintf("templates/include/%s", includeFile))
		fileName, _ := strings.CutSuffix(includeFile, ".tmpl")
		entries = append(entries, Entry{outputName: fmt.Sprintf("include/%s", fileName), parsedTemplate: tmplParsed})
	}
	
	for _, curveData := range curvesData {
		for _, entry := range entries {
			outFile := filepath.Join(baseDir, curveData.PackageName, entry.outputName)
			var buf bytes.Buffer
			if entry.outputName == "scalar_field.go" {
				curveData.IsScalar = true
			} else {
				curveData.IsScalar = false
			}
			entry.parsedTemplate.Execute(&buf, curveData)
			create(outFile, &buf)
		}
	}
}

func main() {
	generateFiles()
}
