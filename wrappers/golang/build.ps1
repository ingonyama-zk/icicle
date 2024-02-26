$G2_DEFINED = "OFF"

if ($args.Count -gt 1) {
  $G2_DEFINED = "ON"
}

$BUILD_DIR = (Get-Location).Path + "\..\icicle\build"
$SUPPORTED_CURVES = @("bn254", "bls12_377", "bls12_381", "bw6_761")

if ($args[0] -eq "all") {
  $BUILD_CURVES = $SUPPORTED_CURVES
} else {
  $BUILD_CURVES = @($args[0])
}

Set-Location "../../icicle"

New-Item -ItemType Directory -Path "build" -Force

foreach ($CURVE in $BUILD_CURVES) {
  cmake -DCURVE:$CURVE -DG2_DEFINED:$G2_DEFINED -DCMAKE_BUILD_TYPE:Release -S . -B build
  cmake --build build
}
