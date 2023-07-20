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

func genMainFiles() {
	bn254_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "g1.go"), Templates: []string{"g1.go.tmpl", "imports.go.tmpl"}},
	}

	bls12377_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "g1.go"), Templates: []string{"g1.go.tmpl", "imports.go.tmpl"}},
	}

	bls12381_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "g1.go"), Templates: []string{"g1.go.tmpl", "imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./curves/", bls12377_entries...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./curves/", bn254_entries...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./curves/", bls12381_entries...))

	bn254_g2_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "g2.go"), Templates: []string{"g2.go.tmpl", "imports.go.tmpl"}},
	}

	bls12377_g2_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "g2.go"), Templates: []string{"g2.go.tmpl", "imports.go.tmpl"}},
	}

	bls12381_g2_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "g2.go"), Templates: []string{"g2.go.tmpl", "imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./curves/", bls12377_g2_entries...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./curves/", bn254_g2_entries...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./curves/", bls12381_g2_entries...))
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

	h_msm_bn254 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "include", "msm.h"), Templates: []string{"msm.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	h_msm_bls12377 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "include", "msm.h"), Templates: []string{"msm.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	h_msm_bls12381 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "include", "msm.h"), Templates: []string{"msm.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./hfiles/", h_msm_bls12377...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./hfiles/", h_msm_bn254...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./hfiles/", h_msm_bls12381...))

	h_ntt_bn254 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "include", "ntt.h"), Templates: []string{"ntt.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	h_ntt_bls12377 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "include", "ntt.h"), Templates: []string{"ntt.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	h_ntt_bls12381 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "include", "ntt.h"), Templates: []string{"ntt.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./hfiles/", h_ntt_bls12377...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./hfiles/", h_ntt_bn254...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./hfiles/", h_ntt_bls12381...))

	ve_mod_mult_h_bn254 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "include", "ve_mod_mult.h"), Templates: []string{"ve_mod_mult.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	ve_mod_mult_h_bls12377 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "include", "ve_mod_mult.h"), Templates: []string{"ve_mod_mult.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	ve_mod_mult_ht_bls12381 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "include", "ve_mod_mult.h"), Templates: []string{"ve_mod_mult.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./hfiles/", ve_mod_mult_h_bls12377...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./hfiles/", ve_mod_mult_h_bn254...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./hfiles/", ve_mod_mult_ht_bls12381...))

	c_api_bn254 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "include", "c_api.h"), Templates: []string{"c_api.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	c_api_bls12377 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "include", "c_api.h"), Templates: []string{"c_api.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	c_api_bls12381 := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "include", "c_api.h"), Templates: []string{"c_api.h.tmpl", "../curves/imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./hfiles/", c_api_bls12377...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./hfiles/", c_api_bn254...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./hfiles/", c_api_bls12381...))
}

func genTestFiles() {
	// G1 TESTS
	bn254_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "g1_test.go"), Templates: []string{"g1_test.go.tmpl", "imports.go.tmpl"}},
	}

	bls12377_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "g1_test.go"), Templates: []string{"g1_test.go.tmpl", "imports.go.tmpl"}},
	}

	bls12381_entries := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "g1_test.go"), Templates: []string{"g1_test.go.tmpl", "imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./curves/", bls12377_entries...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./curves/", bn254_entries...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./curves/", bls12381_entries...))

	// G2 TESTS
	bn254_entries_g2_test := []bavard.Entry{
		{File: filepath.Join(baseDir, "bn254", "g2_test.go"), Templates: []string{"g2_test.go.tmpl", "imports.go.tmpl"}},
	}

	bls12377_entries_g2_test := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12377", "g2_test.go"), Templates: []string{"g2_test.go.tmpl", "imports.go.tmpl"}},
	}

	bls12381_entries_g2_test := []bavard.Entry{
		{File: filepath.Join(baseDir, "bls12381", "g2_test.go"), Templates: []string{"g2_test.go.tmpl", "imports.go.tmpl"}},
	}

	assertNoError(bgen.Generate(config.BLS_12_377, config.BLS_12_377.PackageName, "./curves/", bls12377_entries_g2_test...))
	assertNoError(bgen.Generate(config.BN_254, config.BN_254.PackageName, "./curves/", bn254_entries_g2_test...))
	assertNoError(bgen.Generate(config.BLS_12_381, config.BLS_12_381.PackageName, "./curves/", bls12381_entries_g2_test...))
}

func main() {
	//genMainFiles()
	genTestFiles()
}

func assertNoError(err error) {
	if err != nil {
		fmt.Printf("\n%s\n", err.Error())
		os.Exit(-1)
	}
}
