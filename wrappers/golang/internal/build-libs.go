package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type Config struct {
	MsmDefined       string
	NttDefined       string
	G2Defined        string
	EcnttDefined     string
	ExtField         string
	Hash             string
	Poseidon         string
	Poseidon2        string
	CudaCompilerPath string
	DevMode          string
	BuildCurves      []string
	BuildFields      []string
	CudaBackend      string
	MetalBackend     string
	BuildDir         string
	InstallPath      string
}

var supportedCurves = []string{"bn254", "bls12_377", "bls12_381", "bw6_761", "grumpkin"}
var supportedFields = []string{"babybear", "koalabear"}

func showHelp() {
	fmt.Printf(`Usage: %s [OPTIONS]

Options:
  -curve <string>           Comma-separated list of curves to be built. If "all" is supplied,
                                all curves will be built with any additional curve options.
  -field <string>           Comma-separated list of fields to be built. If "all" is supplied,
                                all fields will be built with any additional field options
  -skip-msm                 Builds the curve library with MSM disabled
  -skip-ntt                 Builds the curve/field library with NTT disabled
  -skip-g2                  Builds the curve library with G2 disabled
  -skip-ecntt               Builds the curve library with ECNTT disabled
  -skip-poseidon            Builds the curve or field library with poseidon hashing disabled
  -skip-poseidon2           Builds the curve or field library with poseidon2 hashing disabled
  -skip-hash                Builds the library with Hashes disabled
  -skip-fieldext            Builds the field library with the extension field disabled
  -cuda <string>            Specifies the branch/commit for CUDA backend, or "local"
  -cuda-version <string>    Specifies the version of CUDA to use
  -metal <string>           Specifies the branch/commit for METAL backend, or "local"
  -install-path <string>    Installation path for built libraries
`, os.Args[0])
}

func main() {
	config := Config{
		MsmDefined:       "ON",
		NttDefined:       "ON",
		G2Defined:        "ON",
		EcnttDefined:     "ON",
		ExtField:         "ON",
		Hash:             "ON",
		Poseidon:         "ON",
		Poseidon2:        "ON",
		CudaCompilerPath: "/usr/local/cuda/bin/nvcc",
		DevMode:          "OFF",
		CudaBackend:      "OFF",
		MetalBackend:     "OFF",
	}

	// Setup command line flags
	curves := flag.String("curve", "", "Comma-separated list of curves")
	fields := flag.String("field", "", "Comma-separated list of fields")
	skipMsm := flag.Bool("skip-msm", false, "Skip MSM")
	skipNtt := flag.Bool("skip-ntt", false, "Skip NTT")
	skipG2 := flag.Bool("skip-g2", false, "Skip G2")
	skipEcntt := flag.Bool("skip-ecntt", false, "Skip ECNTT")
	skipHash := flag.Bool("skip-hash", false, "Skip Hash")
	skipPoseidon := flag.Bool("skip-poseidon", false, "Skip Poseidon")
	skipPoseidon2 := flag.Bool("skip-poseidon2", false, "Skip Poseidon2")
	skipFieldext := flag.Bool("skip-fieldext", false, "Skip Field Extension")
	cudaBackend := flag.String("cuda", "OFF", "CUDA backend")
	cudaVersion := flag.String("cuda-version", "", "CUDA version")
	metalBackend := flag.String("metal", "OFF", "Metal backend")
	installPath := flag.String("install-path", "", "Installation path")
	help := flag.Bool("help", false, "Show help message")

	flag.Parse()

	if *help {
		showHelp()
		return
	}

	// Process flags
	if *skipMsm {
		config.MsmDefined = "OFF"
	}
	if *skipNtt {
		config.NttDefined = "OFF"
	}
	if *skipG2 {
		config.G2Defined = "OFF"
	}
	if *skipEcntt {
		config.EcnttDefined = "OFF"
	}
	if *skipHash {
		config.Hash = "OFF"
	}
	if *skipPoseidon {
		config.Poseidon = "OFF"
	}
	if *skipPoseidon2 {
		config.Poseidon2 = "OFF"
	}
	if *skipFieldext {
		config.ExtField = "OFF"
	}

	// Set CUDA and Metal backends
	config.CudaBackend = *cudaBackend
	config.MetalBackend = *metalBackend

	// Set CUDA compiler path if version specified
	if *cudaVersion != "" {
		config.CudaCompilerPath = fmt.Sprintf("/usr/local/cuda-%s/bin/nvcc", *cudaVersion)
	}

	// Set build and install paths
	pwd, _ := os.Getwd()
	defaultPath := filepath.Join(pwd, "../../../icicle/build")
	config.BuildDir = os.Getenv("ICICLE_BUILD_DIR")
	if config.BuildDir == "" {
		config.BuildDir = defaultPath
	}

	if *installPath != "" {
		config.InstallPath = *installPath
	} else {
		config.InstallPath = os.Getenv("ICICLE_INSTALL_PATH")
		if config.InstallPath == "" {
			config.InstallPath = defaultPath
		}
	}

	// Process curves
	if *curves == "all" {
		config.BuildCurves = supportedCurves
	} else if *curves != "" {
		config.BuildCurves = strings.Split(*curves, ",")
	}

	// Process fields
	if *fields == "all" {
		config.BuildFields = supportedFields
	} else if *fields != "" {
		config.BuildFields = strings.Split(*fields, ",")
	}

	// Change to icicle directory and create build directory
	if err := os.Chdir("../../../icicle"); err != nil {
		fmt.Printf("Error changing directory: %v\n", err)
		os.Exit(1)
	}
	os.MkdirAll("build", 0755)
	os.Remove(filepath.Join(config.BuildDir, "CMakeCache.txt"))

	// Build curves
	for _, curve := range config.BuildCurves {
		fmt.Println("Building curve", curve, "...")
		if err := buildCurve(config, curve); err != nil {
			fmt.Printf("Error building curve %s: %v\n", curve, err)
			os.Exit(1)
		}
		fmt.Println("Finished building curve", curve, "✅")
	}

	// Remove CMakeCache.txt before building fields
	os.Remove(filepath.Join(config.BuildDir, "CMakeCache.txt"))

	// Build fields
	for _, field := range config.BuildFields {
		fmt.Println("Building field", field, "...")
		if err := buildField(config, field); err != nil {
			fmt.Printf("Error building field %s: %v\n", field, err)
			os.Exit(1)
		}
		fmt.Println("Finished building field", field, "✅")
	}

	// Final hash build if enabled
	if config.Hash == "ON" {
		fmt.Println("Building hash ...")
		if err := buildHash(config); err != nil {
			fmt.Printf("Error building hash: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Finished building hash ✅")
	}
}

func buildCurve(config Config, curve string) error {
	// Write build config
	f, err := os.Create("build_config.txt")
	if err != nil {
		return err
	}
	fmt.Fprintf(f, "CURVE=%s\nMSM=%s\nNTT=%s\nECNTT=%s\nG2=%s\nDEVMODE=%s\n",
		curve, config.MsmDefined, config.NttDefined, config.EcnttDefined,
		config.G2Defined, config.DevMode)
	f.Close()

	// Run cmake configure
	cmakeArgs := []string{
		fmt.Sprintf("-DCMAKE_CUDA_COMPILER=%s", config.CudaCompilerPath),
		fmt.Sprintf("-DCUDA_BACKEND=%s", config.CudaBackend),
		fmt.Sprintf("-DMETAL_BACKEND=%s", config.MetalBackend),
		fmt.Sprintf("-DCURVE=%s", curve),
		fmt.Sprintf("-DMSM=%s", config.MsmDefined),
		fmt.Sprintf("-DNTT=%s", config.NttDefined),
		fmt.Sprintf("-DG2=%s", config.G2Defined),
		fmt.Sprintf("-DECNTT=%s", config.EcnttDefined),
		fmt.Sprintf("-DPOSEIDON=%s", config.Poseidon),
		fmt.Sprintf("-DPOSEIDON2=%s", config.Poseidon2),
		fmt.Sprintf("-DHASH=%s", config.Hash),
		"-DCMAKE_BUILD_TYPE=Release",
		fmt.Sprintf("-DCMAKE_INSTALL_PREFIX=%s", config.InstallPath),
		"-S", ".", "-B", "build",
	}

	cmd := exec.Command("cmake", cmakeArgs...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cmake configure failed: %v\n%s", err, out)
	}

	// Run cmake build and install
	cmd = exec.Command("cmake", "--build", "build", "--target", "install")
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cmake build failed: %v\n%s", err, out)
	}

	os.Remove("build_config.txt")
	return nil
}

func buildField(config Config, field string) error {
	// Write build config
	f, err := os.Create("build_config.txt")
	if err != nil {
		return err
	}
	fmt.Fprintf(f, "FIELD=%s\nNTT=%s\nDEVMODE=%s\nEXT_FIELD=%s\n",
		field, config.NttDefined, config.DevMode, config.ExtField)
	f.Close()

	// Run cmake configure
	cmakeArgs := []string{
		fmt.Sprintf("-DCMAKE_CUDA_COMPILER=%s", config.CudaCompilerPath),
		fmt.Sprintf("-DCUDA_BACKEND=%s", config.CudaBackend),
		fmt.Sprintf("-DMETAL_BACKEND=%s", config.MetalBackend),
		fmt.Sprintf("-DFIELD=%s", field),
		fmt.Sprintf("-DNTT=%s", config.NttDefined),
		fmt.Sprintf("-DEXT_FIELD=%s", config.ExtField),
		fmt.Sprintf("-DPOSEIDON=%s", config.Poseidon),
		fmt.Sprintf("-DPOSEIDON2=%s", config.Poseidon2),
		fmt.Sprintf("-DHASH=%s", config.Hash),
		"-DCMAKE_BUILD_TYPE=Release",
		fmt.Sprintf("-DCMAKE_INSTALL_PREFIX=%s", config.InstallPath),
		"-S", ".", "-B", "build",
	}

	cmd := exec.Command("cmake", cmakeArgs...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cmake configure failed: %v\n%s", err, out)
	}

	// Run cmake build and install
	cmd = exec.Command("cmake", "--build", "build", "--target", "install")
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cmake build failed: %v\n%s", err, out)
	}

	os.Remove("build_config.txt")
	return nil
}

func buildHash(config Config) error {
	// Write build config
	f, err := os.Create("build_config.txt")
	if err != nil {
		return err
	}
	fmt.Fprintf(f, "DEVMODE=%s\n", config.DevMode)
	f.Close()

	// Run cmake configure
	cmakeArgs := []string{
		fmt.Sprintf("-DCMAKE_CUDA_COMPILER=%s", config.CudaCompilerPath),
		fmt.Sprintf("-DCUDA_BACKEND=%s", config.CudaBackend),
		fmt.Sprintf("-DMETAL_BACKEND=%s", config.MetalBackend),
		fmt.Sprintf("-DHASH=%s", config.Hash),
		"-DCMAKE_BUILD_TYPE=Release",
		fmt.Sprintf("-DCMAKE_INSTALL_PREFIX=%s", config.InstallPath),
		"-S", ".", "-B", "build",
	}

	cmd := exec.Command("cmake", cmakeArgs...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cmake configure failed: %v\n%s", err, out)
	}

	// Run cmake build and install
	cmd = exec.Command("cmake", "--build", "build", "--target", "install")
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cmake build failed: %v\n%s", err, out)
	}

	os.Remove("build_config.txt")
	return nil
}
