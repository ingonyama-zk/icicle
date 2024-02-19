package main

import (
	"bytes"
	"fmt"
	"io"
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

type Curve struct {
	Curve        string
	ScalarLimbsNum     int
	BaseLimbsNum       int
	G2BaseLimbsNum     int
	IsG2 bool
}

var bn254 = Curve{
	Curve:        	"bn254Temp",
	ScalarLimbsNum: 8,
	BaseLimbsNum:   8,
	G2BaseLimbsNum: 16,
	IsG2: false,
}

type Entry struct {
	outputName string
	parsedTemplate *template.Template
}

func genMainFiles() {
	curvesData := []Curve{bn254}
	var entries []Entry

	tmpl := template.New("msm.go.tmpl")
	tmplParsed, _ := tmpl.ParseFiles("templates/msm.go.tmpl")
	entries = append(entries, Entry{outputName: "msm.go", parsedTemplate: tmplParsed})

	
	for _, curveData := range curvesData {
		for _, entry := range entries {
			outFile := filepath.Join(baseDir, curveData.Curve, entry.outputName)
			var buf bytes.Buffer
			entry.parsedTemplate.Execute(&buf, curveData)
			create(outFile, &buf)
		}
	}
}


func main() {
	fmt.Println("Generating files")
	genMainFiles()
}
