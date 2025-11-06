#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cassert>
#include <cmath>
// #include <omp.h>
#include "include/shard/hnswlib/hnswlib.h"

std::vector<std::vector<float>> load_fvecs(const std::string& filename, int& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("File open error: " + filename);

    std::vector<std::vector<float>> data;
    while (true) {
        int d;
        input.read((char*)&d, 4);
        if (input.eof()) break;
        if (dim == -1) dim = d;
        else assert(dim == d);

        std::vector<float> vec(d);
        input.read((char*)vec.data(), d * 4);
        data.push_back(vec);
    }
    return data;
}

std::vector<std::vector<int>> load_ivecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("File open error: " + filename);

    std::vector<std::vector<int>> data;
    while (true) {
        int d;
        input.read((char*)&d, 4);
        if (input.eof()) break;
        std::vector<int> vec(d);
        input.read((char*)vec.data(), d * 4);
        data.push_back(vec);
    }
    return data;
}

int main() {
    int dim = -1;
    const std::string base_path = "data/sift/sift_base.fvecs";
    const std::string query_path = "data/sift/sift_query.fvecs";
    const std::string gt_path = "data/sift/sift_groundtruth.ivecs";

    int K = 100;  // 1-NN search
    size_t M = 32;
    size_t ef_construction = 200;
    size_t ef_search = 100;

    // Load data
    auto base_data = load_fvecs(base_path, dim);
    auto query_data = load_fvecs(query_path, dim);
    auto groundtruth = load_ivecs(gt_path);
    size_t N = base_data.size();
    size_t Q = query_data.size();

    std::cout << "Loaded base: " << N << ", query: " << Q << ", dim: " << dim << std::endl;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, N, M, ef_construction);

    // Insert with timing
    auto t0 = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        index.addPoint(base_data[i].data(), i);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double insert_time = std::chrono::duration<double>(t1 - t0).count();
    double throughput = N / insert_time;
    std::cout << "[Insert] Throughput: " << throughput << " QPS\n";

    index.setEf(ef_search);

    // Query with timing and compute recall
 // Top-100

    int correct = 0;
    t0 = std::chrono::high_resolution_clock::now();

    // #pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < Q; ++i) {
        auto result = index.searchKnn(query_data[i].data(), K);

        // Ground truth 的前 K 个，作为 set 做快速命中检查
        std::unordered_set<int> gt_set(groundtruth[i].begin(), groundtruth[i].begin() + K);

        while (!result.empty()) {
            int retrieved_id = result.top().second;
            result.pop();
            if (gt_set.count(retrieved_id)) {
                ++correct;
            }
        }
    }
    t1 = std::chrono::high_resolution_clock::now();

    double total_query_time = std::chrono::duration<double>(t1 - t0).count();
    double avg_latency_ms = (total_query_time / Q) * 1000.0;
    double recall = static_cast<double>(correct) / (Q * K);  // 因为每个 query 有 K 个正确答案

    std::cout << "[Query ] Latency: " << avg_latency_ms << " ms/query\n"
            << "[Query ] Recall@100: " << recall << "\n";

    return 0;
}
