build_verification:
	mkdir -p work
	g++ -o work/test_verification -Intt.cpp -Ikernel_ntt.cpp -Iextern.cpp -Iextern_ecntt.cpp -I ectern_extension.cpp -I../../include tests/verification.cpp -std=c++17

build_extern:
	g++ -o work/test_verification -I../../include extern.cpp -std=c++17


extern.o: extern.cpp
    g++ -std=c++17 -std=gnu++11 -c -o $@ $< -I../../include


test_verification: build_verification
	work/test_verification

build_verification_ecntt:
	mkdir -p work
	g++ -o work/test_verification_ecntt -I. -I../../include tests/verification.cpp -std=c++17 -DECNTT

test_verification_ecntt: build_verification_ecntt
	work/test_verification_ecntt
