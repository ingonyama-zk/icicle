CUDA_ROOT_DIR = /usr/local/cuda
NVCC = $(CUDA_ROOT_DIR)/bin/nvcc
CFLAGS = -Xcompiler -fPIC -std=c++17
LDFLAGS = -shared
FEATURES = -DG2_DEFINED

TARGET_BN254 = bn254
TARGET_BLS12_381 = bls12381
TARGET_BLS12_377 = bls12377

VPATH = ../icicle/curves/bn254:../icicle/curves/bls12_377:../icicle/curves/bls12_381

SRCS_BN254 = lde.cu msm.cu projective.cu ve_mod_mult.cu
SRCS_BLS12_381 = lde.cu msm.cu projective.cu ve_mod_mult.cu poseidon.cu
SRCS_BLS12_377 = lde.cu msm.cu projective.cu ve_mod_mult.cu

all: $(TARGET_BN254) $(TARGET_BLS12_381) $(TARGET_BLS12_377)

$(TARGET_BN254): 
	$(NVCC) $(FEATURES) $(CFLAGS) $(LDFLAGS) $(addprefix ../icicle/curves/bn254/, $(SRCS_BN254)) -o $@

$(TARGET_BLS12_381):
	$(NVCC) $(FEATURES) $(CFLAGS) $(LDFLAGS) $(addprefix ../icicle/curves/bls12_381/, $(SRCS_BLS12_381)) -o $@

$(TARGET_BLS12_377):
	$(NVCC) $(FEATURES) $(CFLAGS) $(LDFLAGS) $(addprefix ../icicle/curves/bls12_377/, $(SRCS_BLS12_377)) -o $@

clean:
	rm -f $(TARGET_BN254) $(TARGET_BLS12_381) $(TARGET_BLS12_377)
