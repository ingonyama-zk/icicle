package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/config"
)

const (
	baseDir = "../../temp_generated_files/" // wrappers/golang/curves
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

type Entry struct {
	outputName     string
	parsedTemplate *template.Template
}

func toPackage(s string) string {
	return strings.ReplaceAll(s, "-", "")
}

var templateFuncs = template.FuncMap{
	"log":       fmt.Println,
	"toLower":   strings.ToLower,
	"toUpper":   strings.ToUpper,
	"toPackage": toPackage,
} 

func generateFiles() {
	parseTemplateFile := func(tmplPath string) Entry {
		tmplName := tmplPath[strings.LastIndex(tmplPath, "/")+1:]
		tmpl := template.New(tmplName).Funcs(templateFuncs)
		tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplPath))
		if err != nil {
			panic(err)
		}
		fileName, ok := strings.CutSuffix(tmplName, ".tmpl")
		if !ok {
			panic(".tmpl suffix not found")
		}
		
		return Entry{outputName: fileName, parsedTemplate: tmplParsed}
	}

	fmt.Println("Generating files")
	
	// generate fields
	fieldTemplates := []string{
		"field.go.tmpl",
		"field_test.go.tmpl",
	}

	fieldOperationsTemplates := []string{
		"ntt.go.tmpl",
		"ntt_test.go.tmpl",
		"vec_ops_test.go.tmpl",
		"vec_ops.go.tmpl",
	}

	fieldGeneralTemplates := []string{
		"helpers_test.go.tmpl",
		"main.go.tmpl",
	}

	for _, field := range config.Fields {
		fieldsDir := filepath.Join(baseDir, "fields", field.PackageName)

		for _, fieldTemplate := range fieldTemplates {
			entry := parseTemplateFile("fields/" + fieldTemplate)
			fileName := entry.outputName
			outFile := filepath.Join(fieldsDir, fileName)

			var buf bytes.Buffer
			data := struct {
				FieldData config.FieldData
				FieldPrefix string
				IsScalar bool
				NUM_LIMBS int
			}{
				field,
				"",
				true,
				field.LimbsNum,
			}
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}

		for _, fieldOperation := range fieldOperationsTemplates {
			entry := parseTemplateFile("fields/" + fieldOperation)
			fileName := entry.outputName
			outFile := filepath.Join(fieldsDir, fileName)

			var buf bytes.Buffer
			data := struct {
				FieldData config.FieldData
				FieldPrefix string
			}{
				field,
				"",
			}
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}
		
		for _, fieldGeneral := range fieldGeneralTemplates {
			entry := parseTemplateFile("fields/" + fieldGeneral)
			fileName := strings.Replace(entry.outputName, "main", field.Field, 1)
			outFile := filepath.Join(fieldsDir, fileName)

			var buf bytes.Buffer
			data := struct {
				FieldData config.FieldData
			}{
				field,
			}
			if strings.Contains(fileName, "ecntt") {
				outFile = filepath.Join(outFile, "ecntt")
			}
			if data.IsG2 {
				outFile = filepath.Join(outFile, "g2")
			}
			outFile = filepath.Join(outFile, fileName)
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}
	}

	curveTemplateFiles := []string{
		"msm.go.tmpl",
		"msm_test.go.tmpl",
		"curve_test.go.tmpl",
		"curve.go.tmpl",
	}

	curveGeneralTemplates := []string{
		"helpers_test.go.tmpl",
		"main.go.tmpl",
	}

	for _, curve := range config.Curves {
		curvesDir := filepath.Join(baseDir, "curves", curve.PackageName)

		for _, curveTemplate := range curveTemplateFiles {
			entry := parseTemplateFile("curves/" + curveTemplate)
			fileName := entry.outputName
			
			for _, isG2 := range []bool{false, true} {
				fileNamePrefix := ""
				if isG2 { 
					fileNamePrefix = "g2/g2_"
				}
				outFile := filepath.Join(curvesDir, fileNamePrefix + fileName)
				var buf bytes.Buffer
				data := struct {
					CurveData config.CurveData
					IsG2 bool
					IsMock bool
				}{
					curve,
					isG2,
					false,
				}
				entry.parsedTemplate.Execute(&buf, data)
				create(outFile, &buf)
			}
		}

		for _, curveGeneral := range curveGeneralTemplates {
			entry := parseTemplateFile("curves/" + curveGeneral)
			fileName := strings.Replace(entry.outputName, "main", curve.Curve, 1)
			outFile := filepath.Join(curvesDir, fileName)

			var buf bytes.Buffer
			data := struct {
				CurveData config.CurveData
				FieldPrefix string
			}{
				curve,
				"",
			}
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}
	}
	// 	for _, entry := range curveEntries {
	// 		fileName := strings.Replace(entry.outputName, "main", curve.Curve, 1)
	// 		curveDir := filepath.Join(baseDir, "curves", curve.PackageName)
	// 		if (strings.Contains(fileName, "msm") || strings.Contains(fileName, "curve")) && curve.G2LimbsNum > 0 {
				
	// 		}
	// 		outFile := filepath.Join(curveDir, fileName)
			
	// 		var buf bytes.Buffer
	// 		data := struct {
	// 			config.CurveData
	// 			IsScalar bool
	// 			IsG2     bool
	// 			IsMock   bool
	// 		}{
	// 			curve,
	// 			strings.Contains(fileName, "scalar_field"),
	// 			strings.Contains(fileName, "g2"),
	// 			false,
	// 		}
	// 		entry.parsedTemplate.Execute(&buf, data)
	// 		create(outFile, &buf)
	// 	}
	// }




	// ---------





	// templateFiles := []string{
	// 	"main.go.tmpl",
	// 	"msm.go.tmpl",
	// 	"msm_test.go.tmpl",
	// 	"ntt.go.tmpl",
	// 	"ntt_test.go.tmpl",
	// 	"curve_test.go.tmpl",
	// 	"curve.go.tmpl",
	// 	"vec_ops_test.go.tmpl",
	// 	"vec_ops.go.tmpl",
	// 	"helpers_test.go.tmpl",
	// }

	// for _, tmplName := range templateFiles {
	// 	tmpl := template.New(tmplName).Funcs(funcs)
	// 	tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName))
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	fileName, ok := strings.CutSuffix(tmplName, ".tmpl")
	// 	if !ok {
	// 		panic(err)
	// 	}
	// 	entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
	// }

	// templateG2Files := []string{
	// 	"msm.go.tmpl",
	// 	"msm_test.go.tmpl",
	// 	"curve_test.go.tmpl",
	// 	"curve.go.tmpl",
	// }

	// for _, tmplName := range templateG2Files {
	// 	tmpl := template.New(tmplName).Funcs(funcs)
	// 	tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName))
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	rawFileName, _ := strings.CutSuffix(tmplName, ".tmpl")
	// 	fileName := fmt.Sprintf("g2_%s", rawFileName)
	// 	entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
	// }

	// templateFieldFiles := []string{
	// 	"field.go.tmpl",
	// 	"field_test.go.tmpl",
	// }

	// fieldFilePrefixes := []string{
	// 	"base",
	// 	"g2_base",
	// 	"scalar",
	// }

	// for _, tmplName := range templateFieldFiles {
	// 	tmpl := template.New(tmplName).Funcs(funcs)
	// 	tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", tmplName), "templates/scalar_field.go.tmpl", "templates/scalar_field_test.go.tmpl")
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	fieldFile, _ := strings.CutSuffix(tmplName, ".tmpl")
	// 	for _, fieldPrefix := range fieldFilePrefixes {
	// 		fileName := strings.Join([]string{fieldPrefix, fieldFile}, "_")
	// 		entries = append(entries, Entry{outputName: fileName, parsedTemplate: tmplParsed})
	// 	}
	// }

	// // Header files
	// templateIncludeFiles := []string{
	// 	"curve.h.tmpl",
	// 	"g2_curve.h.tmpl",
	// 	"scalar_field.h.tmpl",
	// 	"msm.h.tmpl",
	// 	"g2_msm.h.tmpl",
	// 	"ntt.h.tmpl",
	// 	"vec_ops.h.tmpl",
	// }

	// for _, includeFile := range templateIncludeFiles {
	// 	tmpl := template.New(includeFile).Funcs(funcs)
	// 	tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/include/%s", includeFile))
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	fileName, _ := strings.CutSuffix(includeFile, ".tmpl")
	// 	entries = append(entries, Entry{outputName: fmt.Sprintf("include/%s", fileName), parsedTemplate: tmplParsed})
	// }

	// for _, curveData := range curvesData {
	// 	for _, entry := range entries {
	// 		fileName := entry.outputName
	// 		if strings.Contains(fileName, "main") {
	// 			fileName = strings.Replace(fileName, "main", curveData.Curve, 1)
	// 		}
	// 		outFile := filepath.Join(baseDir, curveData.PackageName, fileName)
	// 		var buf bytes.Buffer
	// 		data := struct {
	// 			CurveData
	// 			IsScalar bool
	// 			IsG2     bool
	// 			IsMock   bool
	// 		}{
	// 			curveData,
	// 			strings.Contains(fileName, "scalar_field"),
	// 			strings.Contains(fileName, "g2"),
	// 			false,
	// 		}
	// 		entry.parsedTemplate.Execute(&buf, data)
	// 		create(outFile, &buf)
	// 	}
	// }

	// // Generate internal mock field and mock curve files for testing core package
	// internalTemplateFiles := []string{
	// 	"curve_test.go.tmpl",
	// 	"curve.go.tmpl",
	// 	"field_test.go.tmpl",
	// 	"field.go.tmpl",
	// 	"helpers_test.go.tmpl",
	// }

	// for _, internalTemplate := range internalTemplateFiles {
	// 	tmpl := template.New(internalTemplate).Funcs(funcs)
	// 	tmplParsed, err := tmpl.ParseFiles(fmt.Sprintf("templates/%s", internalTemplate))
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	fileName, _ := strings.CutSuffix(internalTemplate, ".tmpl")
	// 	outFile := filepath.Join("../../core/internal/", fileName)

	// 	var buf bytes.Buffer
	// 	data := struct {
	// 		PackageName  string
	// 		BaseLimbsNum int
	// 		IsMock       bool
	// 		IsG2         bool
	// 		IsScalar     bool
	// 	}{
	// 		"internal",
	// 		8,
	// 		true,
	// 		false,
	// 		false,
	// 	}
	// 	tmplParsed.Execute(&buf, data)
	// 	create(outFile, &buf)
	// }
}

func main() {
	generateFiles()
}
