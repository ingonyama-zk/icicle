package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/consensys/bavard"
	config "github.com/ingonyama-zk/icicle/goicicle/templates/curves"
)

const (
	copyrightHolder = "Ingonyama"
	generatedBy     = "Ingonyama"
	copyrightYear   = 2023
	baseDir         = "../curves/"
)

var bgen = bavard.NewBatchGenerator(copyrightHolder, copyrightYear, generatedBy)

func main() {
	bn254_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "bn254.go"), Templates: []string{"curves.go.tmpl", "imports.go.tmpl"}},
	}

	bls12377_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "bls12377.go"), Templates: []string{"curves.go.tmpl", "imports.go.tmpl"}},
	}

	bls12381_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "bls12381.go"), Templates: []string{"curves.go.tmpl", "imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./curves/", bls12377_entries...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./curves/", bn254_entries...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./curves/", bls12381_entries...))
}

func assertNoError(err error) {
	if err != nil {
		fmt.Printf("\n%s\n", err.Error())
		os.Exit(-1)
	}
}
