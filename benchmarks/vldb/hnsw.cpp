#define ENABLE_TIMER

#include "file_util.h"
#include "framework/DynamicExtension.h"
#include "framework/interface/Record.h"
#include "query/knn.h"
#include "shard/Hnsw.h"
#include "standard_benchmarks.h"


#include "../include/shard/hnswlib/hnswlib.h"
#include <gsl/gsl_rng.h>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <queue>

typedef ANNRec Rec;

void usage(char *progname) {
    fprintf(stderr, "%s reccnt datafile queryfile\n", progname);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    // --- 1. 参数解析和数据加载 ---
    size_t n = atol(argv[1]);
    std::string d_fname = std::string(argv[2]);
    std::string q_fname = std::string(argv[3]);

    // HNSW 参数
    const int M = 32;
    const int ef_construction = 200;
    const int ef_search = 100;

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);

    fprintf(stderr, "[I] Reading data file...\n");
    auto data = read_binary_vector_file<Rec>(d_fname, n);

    fprintf(stderr, "[I] Reading Queries\n");
    // 注意：这里读取的查询类型是 Q::Parameters，即 {point, k}
    auto queries = read_binary_knn_queries<de::knn::Query<de::HNSW<Rec>>::Parameters>(q_fname, 1000, 100);

    // --- 2. 构建 HNSW 索引 ---
    fprintf(stderr, "[I] Building pure HNSW index...\n");
    const size_t dim = 128; // 假设 Rec 提供了 get_dim()
    hnswlib::L2SpaceI space(dim);
    hnswlib::HierarchicalNSW<int> hnsw_index(&space, n, M, ef_construction);

    // --- 3. 模拟删除 ---
    fprintf(stderr, "[I] Generating delete vector and marking deletes...\n");
    std::vector<size_t> to_delete;
    to_delete.reserve(n * delete_proportion);
    for (size_t i = 0; i < n; ++i) {
        if (gsl_rng_uniform(rng) <= delete_proportion) {
            to_delete.push_back(i);
        }
    }

    for (size_t i = 0; i < 100000; ++i) {
        hnsw_index.addPoint(data[i].data, i);
    }

    TIMER_INIT();
    TIMER_START();
    int delete_idx = 0;
    for (int i = 100000; i < n; ++i) {
        if (gsl_rng_uniform(rng) <= delete_proportion && to_delete[delete_idx] <= i) {
            // hnsw_index.markDelete(to_delete[delete_idx]);
            // delete_idx++;
        } else {
            hnsw_index.addPoint(data[i].data, i);
        }
    }
    TIMER_STOP();
    auto build_latency = TIMER_RESULT();
    size_t insert_throughput = (size_t)((double)n / (double)build_latency * 1e9);

    // --- 4. HNSW 查询基准测试 ---
    fprintf(stderr, "[I] Running Pure HNSW Query Benchmark\n");
    hnsw_index.setEf(ef_search);

    TIMER_START();
    for (const auto& qp : queries) {
        // 执行查询，但我们暂时不处理结果，只为计时
        hnsw_index.searchKnn(qp.point.data, qp.k);
    }
    TIMER_STOP();
    auto query_latency = TIMER_RESULT() / queries.size();

    // auto hnsw_mem_usage = hnsw_index.getMemoryUsage();

    // 打印与你的框架格式一致的结果
    // insert_throughput | query_latency | memory_usage
    fprintf(stdout, "%ld\t%ld\t%ld\tN/A\tN/A\n", insert_throughput, query_latency, 0);

    // =================================================================
    //               计算 Recall (召回率) 的代码部分
    // =================================================================
    fprintf(stderr, "\n[I] Calculating recall for pure HNSW...\n");

    // 1. 准备“真实”的数据集 (过滤掉被删除的节点)
    std::unordered_set<size_t> delete_indices(to_delete.begin(), to_delete.end());
    std::vector<Rec> ground_truth_data;
    ground_truth_data.reserve(n - delete_indices.size());
    std::vector<size_t> ground_truth_labels; // 同时保存原始标签
    ground_truth_labels.reserve(n - delete_indices.size());

    for (size_t i = 0; i < data.size(); ++i) {
        if (delete_indices.find(i) == delete_indices.end()) {
            ground_truth_data.push_back(data[i]);
            ground_truth_labels.push_back(i);
        }
    }
    fprintf(stderr, "[I] Ground truth data size (after deletes): %zu\n", ground_truth_data.size());

    // 2. 重新执行 HNSW 查询以获取结果
    std::vector<std::vector<hnswlib::labeltype>> hnsw_results;
    hnsw_results.reserve(queries.size());
    for (const auto& qp : queries) {
        auto result_queue = hnsw_index.searchKnn(qp.point.data, qp.k);
        std::vector<hnswlib::labeltype> result_labels;
        result_labels.resize(result_queue.size());
        for (int j = result_queue.size() - 1; j >= 0; --j) {
            result_labels[j] = result_queue.top().second;
            result_queue.pop();
        }
        hnsw_results.push_back(result_labels);
    }
    
    // 3. 为每个查询计算真值并比较
    double total_recall = 0.0;
    #pragma omp parallel for reduction(+:total_recall)
    for (size_t i = 0; i < queries.size(); ++i) {
        // --- HNSW 结果放入集合 ---
        std::unordered_set<hnswlib::labeltype> index_result_set(hnsw_results[i].begin(), hnsw_results[i].end());

        // --- 暴力计算真值 ---
        const auto& qp = queries[i];
        const Rec& query_point = qp.point;
        const size_t k = qp.k;

        auto cmp = [&](const size_t& a_idx, const size_t& b_idx) {
            return query_point.calc_distance(data[a_idx]) < query_point.calc_distance(data[b_idx]);
        };
        std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> top_k_heap(cmp);
        
        for (const auto& label : ground_truth_labels) {
            if (top_k_heap.size() < k) {
                top_k_heap.push(label);
            } else {
                if (query_point.calc_distance(data[label]) < query_point.calc_distance(data[top_k_heap.top()])) {
                    top_k_heap.pop();
                    top_k_heap.push(label);
                }
            }
        }
        
        // --- 比较结果 ---
        int match_count = 0;
        size_t denominator = top_k_heap.size();
        while (!top_k_heap.empty()) {
            if (index_result_set.count(top_k_heap.top())) {
                match_count++;
            }
            top_k_heap.pop();
        }
        
        if (denominator > 0) {
            total_recall += (double)match_count / denominator;
        }
    }

    // 4. 计算并打印平均召回率
    if (!queries.empty()) {
        double average_recall = total_recall / queries.size();
        fprintf(stdout, "[RESULT] Pure HNSW Average Recall: %.4f\n", average_recall);
    }

    gsl_rng_free(rng);
    fflush(stderr);
    fflush(stdout);
    return 0;
}