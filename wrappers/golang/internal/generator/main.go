package main

import (
	"fmt"
	"os"
	"os/exec"
	"path"

	config "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/config"
	curves "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/curves"
	ecntt "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/ecntt"
	fields "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/fields"
	lib_linker "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/lib_linker"
	mock "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/mock"
	msm "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/msm"
	ntt "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/ntt"
	vecops "github.com/ingonyama-zk/icicle/wrappers/golang/internal/generator/vecOps"
)

func generateFiles() {
	fmt.Println("Generating files")

	for _, curve := range config.Curves {
		curveDir := path.Join("curves", curve.PackageName)
		scalarFieldPrefix := "Scalar"
		fields.Generate(curveDir, curve.PackageName, curve.Curve, scalarFieldPrefix, true, curve.ScalarFieldNumLimbs)
		fields.Generate(curveDir, curve.PackageName, curve.Curve, "Base", false, curve.BaseFieldNumLimbs)
		curves.Generate(curveDir, curve.PackageName, curve.Curve, "")
		vecops.Generate(curveDir, curve.Curve, scalarFieldPrefix)
		lib_linker.Generate(curveDir, curve.PackageName, curve.Curve, lib_linker.CURVE, 0)

		if curve.SupportsNTT {
			ntt.Generate(curveDir, "", curve.Curve, scalarFieldPrefix, curve.GnarkImport, 0, true, "", "")
		}

		if curve.SupportsECNTT {
			ecntt.Generate(curveDir, curve.Curve, curve.GnarkImport)
		}

		msm.Generate(curveDir, "msm", curve.Curve, "", curve.GnarkImport)
		if curve.SupportsG2 {
			g2BaseDir := path.Join(curveDir, "g2")
			packageName := "g2"
			fields.Generate(g2BaseDir, packageName, curve.Curve, "G2Base", false, curve.G2FieldNumLimbs)
			curves.Generate(g2BaseDir, packageName, curve.Curve, "G2")
			msm.Generate(curveDir, "g2", curve.Curve, "G2", curve.GnarkImport)
		}
	}

	for _, field := range config.Fields {
		fieldDir := path.Join("fields", field.PackageName)
		scalarFieldPrefix := "Scalar"
		fields.Generate(fieldDir, field.PackageName, field.Field, scalarFieldPrefix, true, field.LimbsNum)
		vecops.Generate(fieldDir, field.Field, scalarFieldPrefix)
		ntt.Generate(fieldDir, "", field.Field, scalarFieldPrefix, field.GnarkImport, field.ROU, true, "", "")
		lib_linker.Generate(fieldDir, field.PackageName, field.Field, lib_linker.FIELD, 0)

		if field.SupportsExtension {
			extensionsDir := path.Join(fieldDir, "extension")
			extensionField := field.Field + "_extension"
			extensionFieldPrefix := "extension"
			fields.Generate(extensionsDir, "extension", extensionField, extensionFieldPrefix, true, field.ExtensionLimbsNum)
			vecops.Generate(extensionsDir, extensionField, extensionFieldPrefix)
			ntt.Generate(fieldDir, "extension", field.Field, scalarFieldPrefix, field.GnarkImport, field.ROU, false, extensionField, extensionFieldPrefix)
			lib_linker.Generate(extensionsDir, "extension", field.Field, lib_linker.FIELD, 1)
		}
	}

	// Mock field and curve files for core
	mock.Generate("core/internal", "internal", "Mock", "MockBase", false, 4)
}

//go:generate go run main.go
func main() {
	generateFiles()

	cmd := exec.Command("gofmt", "-w", "../../")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		fmt.Printf("\n%s\n", err.Error())
		os.Exit(-1)
	}
}
