# Run with `nix-shell cuda-shell.nix`
{ pkgs ? import <nixpkgs> {} }:
let
  gcc10 = pkgs.overrideCC pkgs.stdenv pkgs.gcc10;
in
pkgs.mkShell {
   name = "cuda-env-shell";
   buildInputs = [ gcc10 ]  ++ (with pkgs; [
     git gitRepo gnupg autoconf curl
     procps gnumake util-linux m4 gperf unzip
     cudatoolkit linuxPackages.nvidia_x11
     libGLU libGL
     xorg.libXi xorg.libXmu freeglut
     xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib
     ncurses5
     cmake
   ]);
   cc = gcc10;
   shellHook = ''
     export PATH="$PATH:${pkgs.cudatoolkit}/bin:${pkgs.cudatoolkit}/nvvm/bin"
     export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.linuxPackages.nvidia_x11}/lib
     export CUDA_PATH=${pkgs.cudatoolkit}
     export CPATH="${pkgs.cudatoolkit}/include"
     export LIBRARY_PATH="$LIBRARY_PATH:/lib:${pkgs.linuxPackages.nvidia_x11}/lib"
     export CMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc
     export CXX="${pkgs.gcc10}/bin/c++"
   '';
}
