test_blake2s: test_blake2s.cu blake2s.cu 
	nvcc -o test_blake2s -I. -I../../../include test_blake2s.cu blake2s.cu -g

test_blake2s_hasher: test_blake2s_hasher.cu blake2s.cu 
	nvcc -o test_blake2s_hasher -I. -I../../../include test_blake2s_hasher.cu -g

test_blake2s_tree: test_tree.cu blake2s.cu ../../merkle-tree/merkle.cu
	nvcc -DMERKLE_DEBUG -o test_blake2s_tree -I../../../include test_tree.cu
	./test_blake2s_tree

clear:
	rm test_blake2s test_blake2s_tree test_blake2s_hasher