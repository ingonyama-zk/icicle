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

	bn254_msm_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "msm.go"), Templates: []string{"msm.go.tmpl", "../curves/imports.go.tmpl"}},
	}

	bls12377_msm_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "msm.go"), Templates: []string{"msm.go.tmpl", "../curves/imports.go.tmpl"}},
	}

	bls12381_msm_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "msm.go"), Templates: []string{"msm.go.tmpl", "../curves/imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./msm/", bls12377_msm_entries...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./msm/", bn254_msm_entries...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./msm/", bls12381_msm_entries...))

	bn254_ntt_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "ntt.go"), Templates: []string{"ntt.go.tmpl", "../curves/imports.go.tmpl"}},
	}

	bls12377_ntt_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "ntt.go"), Templates: []string{"ntt.go.tmpl", "../curves/imports.go.tmpl"}},
	}

	bls12381_ntt_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "ntt.go"), Templates: []string{"ntt.go.tmpl", "../curves/imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./ntt/", bls12377_ntt_entries...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./ntt/", bn254_ntt_entries...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./ntt/", bls12381_ntt_entries...))
}

func assertNoError(err error) {
	if err != nil {
		fmt.Printf("\n%s\n", err.Error())
		os.Exit(-1)
	}
}
