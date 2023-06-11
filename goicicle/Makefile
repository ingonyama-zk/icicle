CUDA_ROOT_DIR = /usr/local/cuda
NVCC = $(CUDA_ROOT_DIR)/bin/nvcc
CFLAGS = -arch=sm_30 -std=c++11 -fPIC -g -Wall -Wextra
LDFLAGS = -shared

TARGET_BN254 = libcurves_bn254.so
TARGET_BLS12_381 = libcurves_bls12_381.so
TARGET_BLS12_377 = libcurves_bls12_377.so

SRCS_BN254 = curves_bn254_1.cu curves_bn254_2.cu
SRCS_BLS12_381 = curves_bls12_381_1.cu curves_bls12_381_2.cu
SRCS_BLS12_377 = curves_bls12_377_1.cu curves_bls12_377_2.cu

OBJS_BN254 = $(SRCS_BN254:.cu=.o)
OBJS_BLS12_381 = $(SRCS_BLS12_381:.cu=.o)
OBJS_BLS12_377 = $(SRCS_BLS12_377:.cu=.o)

all: $(TARGET_BN254) $(TARGET_BLS12_381) $(TARGET_BLS12_377)

$(TARGET_BN254): $(OBJS_BN254)
$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET_BLS12_381): $(OBJS_BLS12_381)
$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET_BLS12_377): $(OBJS_BLS12_377)
$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^

.cu.o:
$(NVCC) $(CFLAGS) -c $< -o $@

clean:
rm -f $(TARGET_BN254) $(OBJS_BN254) $(TARGET_BLS12_381) $(OBJS_BLS12_381) $(TARGET_BLS12_377) $(OBJS_BLS12_377)

