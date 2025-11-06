#define ENABLE_TIMER

#include "file_util.h"
#include "framework/DynamicExtension.h"
#include "framework/interface/Record.h"
#include "query/knn.h"
// #include "shard/Efanna.h"
#include "standard_benchmarks.h"

#include <gsl/gsl_rng.h>
#include <unordered_set>

#include "psu-util/timer.h"
#include "shard/Flash.h"
#include "shard/Hnsw.h"
#include "shard/VPTree.h"

typedef ANNRec Rec;

// insert_throughput, query_latency, ext_size, static_latency, static_size
// typedef de::VPTree<Rec, 100, true> Shard;
// 178285  102045640       10537968        120251717       9572816
// typedef de::HNSW<Rec, 32, 200, 10> Shard;
// 12784   3763741 1296926464      361084  516000000
// 1562    4819908 517238464       1097205 516000000
// typedef de::NSG<Rec> Shard;
typedef de::Flash<Rec> Shard;
// 1555    4661957 517238464       1107927 516000000
typedef de::knn::Query<Shard> Q;
typedef de::DynamicExtension<Shard, Q, de::LayoutPolicy::TEIRING,
                             de::DeletePolicy::TAGGING, de::SerialScheduler>
    Ext;
typedef Q::Parameters QP;

void usage(char *progname) {
  fprintf(stderr, "%s reccnt datafile queryfile\n", progname);
}

int main(int argc, char **argv) {  

  if (argc < 4) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  size_t n = atol(argv[1]);
  std::string d_fname = std::string(argv[2]);
  std::string q_fname = std::string(argv[3]);
  std::string gt_fname = std::string(argv[4]);

  auto extension = new Ext(10000, 10000, 8, 0, 64);
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);

  fprintf(stderr, "[I] Reading data file...\n");
  auto data = read_fvecs<Rec, float>(d_fname, n);

  fprintf(stderr, "[I] Generating delete vector\n");
  std::vector<size_t> to_delete(n * delete_proportion);
  size_t j = 0;
  for (size_t i = 0; i < data.size() && j < to_delete.size(); i++) {
    if (gsl_rng_uniform(rng) <= delete_proportion) {
      to_delete[j++] = i;
    }            
  }
  fprintf(stderr, "[I] Reading Queries\n");
  // filename recall@k n
  const int K = 100;
  const int query_num = 100;
  auto queries = read_fvecs_queries<QP, float>(q_fname, K, query_num);

  fprintf(stderr, "[I] Warming up structure...\n");
  size_t warmup = .1 * n;
  size_t delete_idx = 0;
  insert_records<Ext, Rec>(extension, 0, warmup, data, to_delete, delete_idx,
                           false, rng);

  extension->await_next_epoch();

  TIMER_INIT();

  fprintf(stderr, "[I] Running Insertion Benchmark\n");
  TIMER_START();
  insert_records<Ext, Rec>(extension, warmup, data.size(), data, to_delete,
                           delete_idx, true, rng);
  TIMER_STOP();

  auto insert_latency = TIMER_RESULT();
  size_t insert_throughput =
      (size_t)((double)(n - warmup) / (double)insert_latency * 1e9);

  fprintf(stderr, "[I] Running Query Benchmark\n");
  TIMER_START();
  run_queries<Ext, Q>(extension, queries);
  TIMER_STOP();

  auto query_latency = TIMER_RESULT() / queries.size();

  auto shard = extension->create_static_structure();

  fprintf(stderr, "Running Static query tests\n\n");
  TIMER_START();
  run_static_queries<Shard, Q>(shard, queries);
  TIMER_STOP();

  auto static_latency = TIMER_RESULT() / queries.size();

  auto ext_size =
      extension->get_memory_usage() + extension->get_aux_memory_usage();
  auto static_size = shard->get_memory_usage();

  fprintf(stdout, "%ld\t%ld\t%ld\t%ld\t%ld\n", insert_throughput, query_latency,
          ext_size, static_latency, static_size);


    // =================================================================
    //               计算 Recall (召回率) 的代码部分 (已修正)
    // =================================================================
    fprintf(stderr, "\n[I] Calculating recall...\n");

    // 1. 准备“真实”的数据集 (过滤掉被删除的节点)
    // -----------------------------------------------------------------
    // 为了方便查找，我们将 to_delete 数组转换成一个哈希集合
    // std::unordered_set<size_t> delete_indices;
    // // 确保 to_delete 向量是有效的
    // if (!to_delete.empty()) {
    //     delete_indices.insert(to_delete.begin(), to_delete.end());
    // }

    /* 如果有删除的数据点的真值似乎只能暴力计算（？）
    std::vector<Rec> ground_truth_data
    ground_truth_data.reserve(data.size());
    // ground_truth_data.reserve(n - delete_indices.size());

    for (size_t i = 0; i < data.size(); ++i) {
      // 如果当前索引 i 不在待删除的集合中，则将其加入到真实数据集中
      // if (delete_indices.find(i) == delete_indices.end()) {
      ground_truth_data.push_back(data[i]);
      // }
    }
    fprintf(stderr, "[I] Ground truth data size (after deletes): %zu\n",
ground_truth_data.size());

    // 2. 重新执行查询以获取结果
    // -----------------------------------------------------------------
    // extension->query() 返回 std::future<std::vector<R>>
    std::vector<std::future<std::vector<Rec>>> query_futures;
    for (const auto& qp_struct : queries) {
        // qp_struct 是 knn_query_t<Rec> 类型，它有一个成员叫 query_point
        // 我们需要用它来构造 Q::Parameters
        query_futures.push_back(extension->query(QP{qp_struct.point,
qp_struct.k}));
    }

    // 3. 为每个查询计算真值并比较
    // -----------------------------------------------------------------
    double total_recall = 0.0;
#pragma omp parallel for reduction(+ : total_recall)
    for (size_t i = 0; i < queries.size(); ++i) {
        // 获取索引返回的结果，类型是 std::vector<Rec>
        auto index_results = query_futures[i].get();
        // 将结果放入哈希集合中以便快速查找
        std::unordered_set<Rec, de::RecordHash<Rec>>
index_result_set(index_results.begin(), index_results.end());

        // --- 暴力计算真值 ---
        const auto& qp_struct = queries[i];
        const Rec& query_point = qp_struct.point;
        const size_t k = qp_struct.k;

        // 使用一个最小堆来找到 top-k，堆顶是距离最远的点
        auto cmp = [&](const Rec& a, const Rec& b) {
            return query_point.calc_distance(a) < query_point.calc_distance(b);
        };
        std::priority_queue<Rec, std::vector<Rec>, decltype(cmp)>
top_k_heap(cmp);

        for (const auto& point : ground_truth_data) {
            if (top_k_heap.size() < k) {
                top_k_heap.push(point);
            } else {
                // 如果当前点的距离比堆顶（最远点）的距离还小，则替换
                if (query_point.calc_distance(point) <
query_point.calc_distance(top_k_heap.top())) { top_k_heap.pop();
                    top_k_heap.push(point);
                }
            }
        }

        // --- 比较结果并计算当前查询的召回率 ---
        int match_count = 0;
        while (!top_k_heap.empty()) {
            Rec ground_truth_point = top_k_heap.top();
            top_k_heap.pop();

            // 检查真值点是否存在于索引返回的结果集合中
            if (index_result_set.count(ground_truth_point)) {
                match_count++;
            }
        }

        // k值可能大于实际数据量，取较小者作为分母
        size_t denominator = std::min(k, ground_truth_data.size());
        if (denominator == 0) continue;

        double current_recall = (double)match_count / denominator;
        total_recall += current_recall;
    }
*/

    // std::vector<Rec> ground_truth_data =
    //     read_fvecs<Rec, int>(gt_fname, query_num);
    std::vector<std::vector<int>> ground_truth_data(query_num);
    {
      std::ifstream file(gt_fname, std::ios::binary);

      for (size_t i = 0; i < query_num; i++) {
        int32_t dim;
        file.read((char *)&(dim), sizeof(int32_t));
        ground_truth_data[i].resize(dim);
        for (size_t j = 0; j < dim; j++) {
          file.read(reinterpret_cast<char *>(&ground_truth_data[i][j]),
                    sizeof(int));
        }
      }
    }

    int match_count = 0;
    for (int i = 0; i < query_num; ++i) {
      auto &qp_struct = queries[i];
      auto res = extension->query(QP{qp_struct.point, qp_struct.k}).get();
      auto gt = ground_truth_data[i];
      if (i == 1) {
        for (int p = 0; p < K; ++p) {
          int id = res[p].id;
          std::cout << id << " ";
        }
        std::cout << std::endl;
        for (int p = 0; p < K; ++p) {
          int id = gt[p];
          std::cout << id << " ";
        }
        std::cout << std::endl;
      }
      for (int p = 0; p < K; ++p) {
        int id = gt[p]; // 枚举每个真值，查看 res 里面是否有
        for (int q = 0; q < K; ++q) {
          if (id == res[q].id) {
            match_count++;
            break;
          }
        }
      }
    }
    double total_recall = (double)match_count / (query_num * K);
    printf("match_count: %d\n", match_count);
    printf("Recall: %.4f\n", total_recall);

    // double total_recall = 0.0;

    // // #pragma omp parallel for reduction(+ : total_recall)
    // for (size_t i = 0; i < queries.size(); ++i) {
    //   // 获取索引返回的结果，类型是 std::vector<Rec>
    //   auto index_results = query_futures[i].get();
    //   if (i == 0) {
    //     std::cout << index_results.size() << std::endl;
    //     // std::cout << std::strlen(index_results[0].data) << std::endl;
    //   }
    //   // 将结果放入哈希集合中以便快速查找
    //   std::unordered_set<Rec, de::RecordHash<Rec>> index_result_set(
    //       index_results.begin(), index_results.end());

    //   const auto &qp_struct = queries[i];
    //   const Rec &query_point = qp_struct.point;
    //   const size_t k = qp_struct.k;

    //   Rec &ground_truth_point = ground_truth_data[j]; // 我需要取出前 k 个
    //   for (int p = 0; p < k; ++p) {
    //     std::cout << index_results[p].data[0] << std::endl;
    //   }
    //   for (int p = 0; p < k; ++p) {
    //     for (int q = 0; q < k; ++q) {
    //       if (i == 0 && p == 0) {
    //         std::cout << index_results[p].data[0] << " "
    //                   << ground_truth_data[q].data[0] << std::endl;
    //       }
    //       if (index_results[p].data[0] == ground_truth_data[q].data[1]) {
    //         match_count++;
    //         break;
    //       }
    //     }
    //   }
    // }
    // total_recall = (double)match_count / (queries.size() * K);

    // 4. 计算并打印平均召回率
    // -----------------------------------------------------------------
    // if (queries.empty()) {
    //   fprintf(stdout, "[RESULT] Average Recall: N/A (0 queries)\n");
    // } else {
    //   double average_recall = total_recall / queries.size();
    //   fprintf(stdout, "[RESULT] Average Recall: %.4f\n", average_recall);
    // }

    // =================================================================
    //                     计算 Recall 代码结束
    // =================================================================

    gsl_rng_free(rng);
    delete extension;
    fflush(stderr);
    fflush(stdout);
}