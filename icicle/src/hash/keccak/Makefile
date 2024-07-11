test_keccak_tree: test_tree.cu keccak.cu ../../merkle-tree/merkle.cu
	nvcc -DMERKLE_DEBUG -o test_keccak_tree -I../../../include test_tree.cu
	./test_keccak_tree

test_keccak: test.cu keccak.cu 
	nvcc -o test_keccak -I../../../include test.cu
	./test_keccak

clear:
	rm test_keccak test_keccak_tree