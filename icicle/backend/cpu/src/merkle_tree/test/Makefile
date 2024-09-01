build_msm_test:
	mkdir -p build
	g++ -O3 -c ../merkle_tree.cpp -o build/merkle_tree.o -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	g++ -O3 -o build/merkle_test merkle_test.cpp build/merkle_tree.o -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
run_memory_check:
	mkdir -p build
	g++ -O3 -c ../merkle_tree.cpp -o build/merkle_tree.o -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	g++ -O3 -fsanitize=address -g -o build/merkle_test merkle_test.cpp build/merkle_tree.o -I.. -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	./build/merkle_test

build_debug:
	mkdir -p build
	mkdir -p build/generated_data
	g++ -g -o build/merkle_test merkle_test.cpp -I.. -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17

clean_logs:
	rm -rf *.txt
	rm -rf ./build/*.txt