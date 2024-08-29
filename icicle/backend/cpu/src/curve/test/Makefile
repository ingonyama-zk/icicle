curve=4

build_msm_test:
	mkdir -p build
	mkdir -p build/generated_data
	g++ -O3 -o build/msm_test msm_test.cpp -I.. -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17 -DCURVE_ID=$(curve)

run_memory_check:
	mkdir -p build
	mkdir -p build/generated_data
	g++ -O3 -fsanitize=address -fno-omit-frame-pointer -g -o build/msm_test msm_test.cpp -I.. -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17 -DCURVE_ID=$(curve)
	./build/msm_test

build_debug:
	mkdir -p build
	mkdir -p build/generated_data
	g++ -g -o build/msm_test msm_test.cpp -I.. -I../../../../../include -I../../../include -I../../../../../include/icicle/utils/ -std=c++17 -DCURVE_ID=$(curve)

diff_bkts:
	diff buckets_multi.txt buckets_single.txt

trace_idx:
	grep "#$(idx):" trace_bucket_multi.txt > single_trace_multithread.txt
	grep "#$(idx):" trace_bucket_single.txt > single_trace_singlethread.txt

clean_logs:
	rm -rf *.txt
	rm -rf ./build/*.txt

clean_data:
	rm -rf ./build/generated_data/*
	