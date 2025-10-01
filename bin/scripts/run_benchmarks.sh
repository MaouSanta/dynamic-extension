# !/bin/bash

dataset_dir='./data/dataset/'
query_dir='./data/queries/'
bench_dir='./bin/benchmarks/'

binary_vector_datasets=('sift_base.fvecs')
binary_vector_queries=('sift_query.fvecs')

# binary_vector_benchmarks=('vptree_bench_alt' 'hnsw_bench_alt' 'hnsw')
binary_vector_benchmarks=('hnsw_bench_alt')

for bench in ${binary_vector_benchmarks[@]}; do
	for (( i=0; i<1; i++)); do
		dataset=${binary_vector_datasets[$i]}
		query=${binary_vector_queries[$i]}
		printf "%20s\t%30s\t\t" "$bench" "$dataset"
		cnt=10000
		
		# numactl -C1 -m1
		"$bench_dir""$bench" "$cnt" "$dataset_dir""$dataset" "$query_dir""$query"

		if [[ ! $? ]]; then
			printf "ERROR\n"
		fi
	done
done


