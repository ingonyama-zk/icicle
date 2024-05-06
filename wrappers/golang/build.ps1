$G2_DEFINED = "OFF"
$ECNTT_DEFINED = "OFF"
$CUDA_COMPILER_PATH = "/usr/local/cuda/bin/nvcc"
$DEVMODE = "OFF"
$EXT_FIELD = "OFF"
$BUILD_CURVES = @()
$BUILD_FIELDS = @()
$BUILD_HASHES = @()

$SUPPORTED_CURVES = @("bn254", "bls12_377", "bls12_381", "bw6_761", "grumpkin")
$SUPPORTED_FIELDS = @("babybear")
$SUPPORTED_HASHES = @("keccak")

foreach ($arg in $args) {
  $arg_lower = $arg.ToLower()
  switch -Wildcard ($arg_lower) {
    "-cuda_version=*" {
      $cuda_version = $arg.Split("=")[1]
      $CUDA_COMPILER_PATH = "/usr/local/cuda-$cuda_version/bin/nvcc"
    }
    "-ecntt" {
      $ECNTT_DEFINED = "ON"
    }
    "-g2" {
      $G2_DEFINED = "ON"
    }
    "-curve=*" {
      $curve = ($arg_lower -split "=")[1]
      if ($curve -eq "all") {
        $BUILD_CURVES = $SUPPORTED_CURVES
      } else {
        $BUILD_CURVES = @($curve)
      }
    }
    "-field=*" {
      $field = ($arg_lower -split "=")[1]
      if ($field -eq "all") {
        $BUILD_FIELDS = $SUPPORTED_FIELDS
      } else {
        $BUILD_FIELDS = @($field)
      }
    }
    "-field-ext" {
      $EXT_FIELD = "ON"
    }
    "-hash=*" {
      $hash = ($arg_lower -split "=")[1]
      if ($hash -eq "all") {
        $BUILD_HASHES = $SUPPORTED_HASHES
      } else {
        $BUILD_HASHES = @($hash)
      }
    }
    "-devmode" {
      $DEVMODE = "ON"
    }
    "-help" {
        Write-Host "Build script for building ICICLE cpp libraries"
        Write-Host ""
        Write-Host "If more than one curve or more than one field is supplied, the last one supplied will be built"
        Write-Host ""
        Write-Host "USAGE: .\build.ps1 [OPTION...]"
        Write-Host ""
        Write-Host "OPTIONS:"
        Write-Host "  -curve=<curve_name>       The curve that should be built. If \"all\" is supplied,"
        Write-Host "                            all curves will be built with any other supplied curve options"
        Write-Host "  -g2                       Builds the curve lib with G2 enabled"
        Write-Host "  -ecntt                    Builds the curve lib with ECNTT enabled"
        Write-Host "  -field=<field_name>       The field that should be built. If \"all\" is supplied,"
        Write-Host "                            all fields will be built with any other supplied field options"
        Write-Host "  -field-ext                Builds the field lib with the extension field enabled"
        Write-Host "  -hash=<hash>              The name of the hash to build or "all" to build all supported hashes"
        Write-Host "  -devmode                  Enables devmode debugging and fast build times"
        Write-Host "  -cuda_version=<version>   The version of cuda to use for compiling"
        Write-Host ""

        exit 0
    }
    default {
      Write-Host "Unknown argument: $arg"
      exit 1
    }
  }
}

$BUILD_DIR = (Get-Location).Path + "\..\..\icicle\build"

Set-Location "..\..\icicle"
New-Item -ItemType Directory -Path "build" -Force
Remove-Item -Path "$BUILD_DIR\CMakeCache.txt" -ErrorAction SilentlyContinue

foreach ($CURVE in $BUILD_CURVES) {
  Out-File -FilePath "build_config.txt" -InputObject "CURVE=$CURVE" -Append
  Out-File -FilePath "build_config.txt" -InputObject "ECNTT=$ECNTT_DEFINED" -Append
  Out-File -FilePath "build_config.txt" -InputObject "G2=$G2_DEFINED" -Append
  Out-File -FilePath "build_config.txt" -InputObject "DEVMODE=$DEVMODE" -Append
  cmake -DCURVE:STRING=$CURVE -DG2:STRING=$G2_DEFINED -DECNTT:STRING=$ECNTT_DEFINED -DCMAKE_CUDA_COMPILER:STRING=$CUDA_COMPILER_PATH -DDEVMODE:STRING=$DEVMODE -DCMAKE_BUILD_TYPE:STRING=Release -S . -B build
  cmake --build build -j 8
  Remove-Item -Path "build_config.txt"
}

Remove-Item -Path "$BUILD_DIR\CMakeCache.txt" -ErrorAction SilentlyContinue

foreach ($FIELD in $BUILD_FIELDS) {
  Out-File -FilePath "build_config.txt" -InputObject "FIELD=$FIELD" -Append
  Out-File -FilePath "build_config.txt" -InputObject "DEVMODE=$DEVMODE" -Append
  cmake -DFIELD:STRING=$FIELD -DEXT_FIELD:STRING=$EXT_FIELD -DCMAKE_CUDA_COMPILER:STRING=$CUDA_COMPILER_PATH -DDEVMODE:STRING=$DEVMODE -DCMAKE_BUILD_TYPE:STRING=Release -S . -B build
  cmake --build build -j 8
  Remove-Item -Path "build_config.txt"
}

foreach ($HASH in $BUILD_HASHES) {
  Out-File -FilePath "build_config.txt" -InputObject "HASH=$HASH" -Append
  Out-File -FilePath "build_config.txt" -InputObject "DEVMODE=$DEVMODE" -Append
  cmake -DHASH:STRING=$HASH -DCMAKE_CUDA_COMPILER:STRING=$CUDA_COMPILER_PATH -DDEVMODE:STRING=$DEVMODE -DCMAKE_BUILD_TYPE:STRING=Release -S . -B build
  cmake --build build -j 8
  Remove-Item -Path "build_config.txt"
}
