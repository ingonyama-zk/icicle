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

func toCName(s string) string {
	if s == "" {
		return ""
	}
	return strings.ToLower(s) + "_"
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
	"log":        fmt.Println,
	"toLower":    strings.ToLower,
	"toUpper":    strings.ToUpper,
	"toPackage":  toPackage,
	"toCName":    toCName,
	"toConst":    toConst,
	"capitalize": capitalize,
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

	// Field template
	fieldTemplates := []string{
		"field.go.tmpl",
		"field_test.go.tmpl",
	}

	// Templates specific to field operations that are always supported
	fieldOperationsTemplates := []string{
		"vec_ops_test.go.tmpl",
		"vec_ops.go.tmpl",
	}

	// Other templates for fields
	fieldGeneralTemplates := []string{
		"helpers_test.go.tmpl",
		"main.go.tmpl",
	}

	for _, field := range config.Fields {
		fieldsDir := filepath.Join(baseDir, "fields", field.PackageName)
		templates := append(fieldTemplates, append(fieldOperationsTemplates, fieldGeneralTemplates...)...)

		if field.SupportsNTT {
			templates = append(templates, "ntt.go.tmpl", "ntt_test.go.tmpl")
		}

		for _, fieldTemplate := range templates {
			entry := parseTemplateFile("fields/" + fieldTemplate)
			fileName := strings.Replace(entry.outputName, "main", field.Field, 1)
			outFile := filepath.Join(fieldsDir, fileName)

			var buf bytes.Buffer
			data := struct {
				PackageName string
				Field       string
				GnarkImport string
				FieldPrefix string
				IsScalar    bool
				NUM_LIMBS   int
			}{
				field.PackageName,
				field.Field,
				field.GnarkImport,
				"",
				false,
				field.LimbsNum,
			}
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}
	}

	// Templates specific to curve and g2 operations
	curveTemplateFiles := []string{
		"msm.go.tmpl",
		"msm_test.go.tmpl",
		"curve_test.go.tmpl",
		"curve.go.tmpl",
	}

	// Other templates for curves
	curveGeneralTemplates := []string{
		"helpers_test.go.tmpl",
		"main.go.tmpl",
	}

	for _, curve := range config.Curves {
		curvesDir := filepath.Join(baseDir, "curves", curve.PackageName)

		for _, curveTemplate := range append(curveTemplateFiles, curveGeneralTemplates...) {
			entry := parseTemplateFile("curves/" + curveTemplate)
			fileName := strings.Replace(entry.outputName, "main", curve.Curve, 1)

			outFile := filepath.Join(curvesDir, fileName)
			var buf bytes.Buffer
			data := struct {
				PackageName string
				Curve       string
				GnarkImport string
				CurvePrefix string
			}{
				curve.PackageName,
				curve.Curve,
				curve.GnarkImport,
				"",
			}
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}

		// G2 Field
		if curve.SupportsG2 {
			fieldPrefix := "G2"
			for _, curveTemplate := range curveTemplateFiles {
				entry := parseTemplateFile("curves/" + curveTemplate)
				fileName := entry.outputName
				outFile := filepath.Join(curvesDir, "g2/g2_"+fileName)
				var buf bytes.Buffer
				data := struct {
					PackageName string
					CurvePrefix string
				}{
					curve.PackageName,
					fieldPrefix,
				}
				entry.parsedTemplate.Execute(&buf, data)
				create(outFile, &buf)
			}
			for _, curveTemplate := range fieldTemplates {
				entry := parseTemplateFile("fields/" + curveTemplate)
				fileName := entry.outputName
				outFile := filepath.Join(curvesDir, "g2/g2_"+fileName)
				var buf bytes.Buffer
				data := struct {
					PackageName string
					Field       string
					GnarkImport string
					FieldPrefix string
					IsScalar    bool
					NUM_LIMBS   int
				}{
					"g2",
					curve.Curve,
					curve.GnarkImport,
					fieldPrefix,
					false,
					curve.G2FieldNumLimbs,
				}
				entry.parsedTemplate.Execute(&buf, data)
				create(outFile, &buf)
			}
		}

		if curve.SupportsECNTT {
			for _, curveTemplate := range []string{"ecntt.go.tmpl", "ecntt_test.go.tmpl"} {
				entry := parseTemplateFile("curves/" + curveTemplate)
				fileName := entry.outputName
				outFile := filepath.Join(curvesDir, "ecntt/ecntt_"+fileName)
				var buf bytes.Buffer
				data := struct {
					Curve string
				}{
					curve.Curve,
				}
				entry.parsedTemplate.Execute(&buf, data)
				create(outFile, &buf)
			}
		}

		if curve.SupportsPoseidon {

		}

		// Field operations for curve
		fieldGeneralTemplates := fieldOperationsTemplates
		if curve.SupportsNTT {
			fieldGeneralTemplates = append(fieldGeneralTemplates, "ntt.go.tmpl", "ntt_test.go.tmpl")
		}

		for _, fieldTemplate := range fieldGeneralTemplates {
			entry := parseTemplateFile("fields/" + fieldTemplate)
			fileName := entry.outputName
			outFile := filepath.Join(curvesDir, fileName)

			var buf bytes.Buffer
			data := struct {
				PackageName string
				Field       string
				GnarkImport string
				FieldPrefix string
			}{
				curve.PackageName,
				curve.Curve,
				curve.GnarkImport,
				"Scalar",
			}
			entry.parsedTemplate.Execute(&buf, data)
			create(outFile, &buf)
		}

		// Scalar + Base Fields for curve
		for _, isScalar := range []bool{false, true} {
			for _, fieldTemplate := range fieldTemplates {
				entry := parseTemplateFile("fields/" + fieldTemplate)
				fileName := entry.outputName
				fieldPrefix := "base"
				numLimbs := curve.BaseFieldNumLimbs
				if isScalar {
					fieldPrefix = "scalar"
					numLimbs = curve.ScalarFieldNumLimbs
				}
				outFile := filepath.Join(curvesDir, fieldPrefix+"_"+fileName)

				var buf bytes.Buffer
				data := struct {
					PackageName string
					Field       string
					GnarkImport string
					FieldPrefix string
					IsScalar    bool
					NUM_LIMBS   int
				}{
					curve.PackageName,
					curve.Curve,
					curve.GnarkImport,
					capitalize(fieldPrefix),
					isScalar,
					numLimbs,
				}
				entry.parsedTemplate.Execute(&buf, data)
				create(outFile, &buf)
			}
		}
	}

	// TODO: add helpers_test.go to each package including ecntt and g2
	// TODO: header files
	// TODO: mock field and curves
	// TODO: Babybear
	// TODO: update templates for correct code including transpose and removal of generics


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
