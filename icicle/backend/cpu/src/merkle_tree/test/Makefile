build_msm_test:
	mkdir -p build
	g++ -O3 -c ../merkle_tree.cpp -o build/merkle_tree.o -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	g++ -O3 -o build/merkle_test merkle_test.cpp build/merkle_tree.o -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	./build/merkle_test

mem_check:
	mkdir -p build
	g++ -g3 -fsanitize=address -c ../merkle_tree.cpp -o build/merkle_tree.o -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	g++ -g3 -fsanitize=address -o build/merkle_test merkle_test.cpp build/merkle_tree.o -I.. -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	./build/merkle_test

debug:
	mkdir -p build
	g++ -g3 -O0 -fsanitize=address -c ../merkle_tree.cpp -o build/merkle_tree.o -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17
	g++ -g3 -O0 -fsanitize=address -o build/merkle_test merkle_test.cpp build/merkle_tree.o -I.. -I.. -I../../hash -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17

clean_logs:
	rm -rf *.txt
	rm -rf ./build/*.txt

clean_comp:
	rm -rf ./build/*