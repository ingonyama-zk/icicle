package main

import (
	"fmt"
	"os"
	"os/exec"
	"path"

	config "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/config"
	curves "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/curves"
	ecntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/ecntt"
	fields "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/fields"
	lib_linker "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/lib_linker"
	mock "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/mock"
	msm "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/msm"
	ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/ntt"
	poly "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/polynomial"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/poseidon"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/poseidon2"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/tests"
	vecops "github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/generator/vecOps"
)

func generateFiles() {
	fmt.Println("Generating files")

	for _, curve := range config.Curves {
		curveDir := path.Join("curves", curve.PackageName)
		scalarFieldPrefix := "Scalar"
		fields.Generate(curveDir, curve.PackageName, curve.Curve, scalarFieldPrefix, true, curve.ScalarFieldNumLimbs)
		fields.Generate(curveDir, curve.PackageName, curve.Curve, "Base", false, curve.BaseFieldNumLimbs)
		curves.Generate(curveDir, curve.PackageName, curve.Curve, "")
		vecops.Generate(curveDir, curveDir, curve.Curve, scalarFieldPrefix, curve.Curve)

		lib_linker.Generate(curveDir, curve.PackageName, curve.Curve, lib_linker.CURVE, 0)

		if curve.SupportsNTT {
			ntt.Generate(curveDir, "", curve.Curve, scalarFieldPrefix, curve.GnarkImport, true, "", "")
			poly.Generate(curveDir, curve.Curve, scalarFieldPrefix, curve.GnarkImport)
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

		if curve.SupportsPoseidon {
			poseidon.Generate(curveDir, curve.Curve)
		}
		if curve.SupportsPoseidon2 {
			poseidon2.Generate(curveDir, curve.Curve)
		}

		tests.Generate(curveDir, curve.Curve, scalarFieldPrefix, curve.GnarkImport, curve.SupportsNTT)
	}

	for _, field := range config.Fields {
		fieldDir := path.Join("fields", field.PackageName)
		scalarFieldPrefix := "Scalar"
		fields.Generate(fieldDir, field.PackageName, field.Field, scalarFieldPrefix, true, field.LimbsNum)
		vecops.Generate(fieldDir, fieldDir, field.Field, scalarFieldPrefix, field.Field)
		if field.SupportsNTT {
			ntt.Generate(fieldDir, "", field.Field, scalarFieldPrefix, field.GnarkImport, true, "", "")
			poly.Generate(fieldDir, field.Field, scalarFieldPrefix, field.GnarkImport)
		}
		lib_linker.Generate(fieldDir, field.PackageName, field.Field, lib_linker.FIELD, 0)

		if field.SupportsExtension {
			extensionsDir := path.Join(fieldDir, "extension")
			extensionField := field.Field + "_extension"
			extensionFieldPrefix := "Extension"
			fields.Generate(extensionsDir, "extension", extensionField, extensionFieldPrefix, true, field.ExtensionLimbsNum)
			vecops.Generate(extensionsDir, fieldDir, extensionField, extensionFieldPrefix, field.Field)
			if field.SupportsNTT {
				ntt.Generate(fieldDir, "extension", field.Field, scalarFieldPrefix, field.GnarkImport, false, extensionField, extensionFieldPrefix)
			}
		}

		if field.SupportsPoseidon {
			poseidon.Generate(fieldDir, field.Field)
		}

		if field.SupportsPoseidon2 {
			poseidon2.Generate(fieldDir, field.Field)
		}

		tests.Generate(fieldDir, field.Field, scalarFieldPrefix, field.GnarkImport, field.SupportsNTT)
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
