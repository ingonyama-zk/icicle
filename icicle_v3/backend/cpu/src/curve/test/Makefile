build_msm_test:
	mkdir -p build
	mkdir -p build/generated_data
	g++ -O3 -o build/msm_test msm_test.cpp -I.. -I../../../../../include -I../../../include -std=c++17 -DCURVE_ID=1

build_debug:
	mkdir -p build
	mkdir -p build/generated_data
	g++ -g -o build/msm_test msm_test.cpp -I.. -I../../../../../include -I../../../include -std=c++17 -DCURVE_ID=1

diff_bkts:
	diff buckets_multi.txt buckets_single.txt

trace_idx:
	grep "#$(idx):" trace_bucket_multi.txt > single_trace_multithread.txt
	grep "#$(idx):" trace_bucket_single.txt > single_trace_singlethread.txt

clean_logs:
	rm -f *.txt
	