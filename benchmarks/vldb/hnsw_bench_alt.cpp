#define ENABLE_TIMER

#include "file_util.h"
#include "framework/DynamicExtension.h"
#include "framework/interface/Record.h"
#include "query/knn.h"
#include "shard/Hnsw.h"
#include "standard_benchmarks.h"

#include <gsl/gsl_rng.h>
#include <unordered_set>

#include "psu-util/timer.h"

typedef ANNRec Rec;

typedef de::HNSW<Rec, 32, 200, 10> Shard;
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

  auto extension = new Ext(1000000, 1000000, 8, 0, 64);
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);

  fprintf(stderr, "[I] Reading data file...\n");
  auto data = read_binary_vector_file<Rec>(d_fname, n);

  fprintf(stderr, "[I] Generating delete vector\n");
  std::vector<size_t> to_delete(n * delete_proportion);
  size_t j = 0;
  for (size_t i = 0; i < data.size() && j < to_delete.size(); i++) {
    if (gsl_rng_uniform(rng) <= delete_proportion) {
      to_delete[j++] = i;
    }
  }
  fprintf(stderr, "[I] Reading Queries\n");
  auto queries = read_binary_knn_queries<QP>(q_fname, 1000, 100);

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
    std::unordered_set<size_t> delete_indices;
    // 确保 to_delete 向量是有效的
    if (!to_delete.empty()) {
        delete_indices.insert(to_delete.begin(), to_delete.end());
    }
    
    std::vector<Rec> ground_truth_data;
    ground_truth_data.reserve(n - delete_indices.size());

    for (size_t i = 0; i < data.size(); ++i) {
        // 如果当前索引 i 不在待删除的集合中，则将其加入到真实数据集中
        if (delete_indices.find(i) == delete_indices.end()) {
            ground_truth_data.push_back(data[i]);
        }
    }
    fprintf(stderr, "[I] Ground truth data size (after deletes): %zu\n", ground_truth_data.size());

    // 2. 重新执行查询以获取结果
    // -----------------------------------------------------------------
    // extension->query() 返回 std::future<std::vector<R>>
    std::vector<std::future<std::vector<Rec>>> query_futures;
    for (const auto& qp_struct : queries) {
        // qp_struct 是 knn_query_t<Rec> 类型，它有一个成员叫 query_point
        // 我们需要用它来构造 Q::Parameters
        query_futures.push_back(extension->query(QP{qp_struct.point, qp_struct.k}));
    }

    // 3. 为每个查询计算真值并比较
    // -----------------------------------------------------------------
    double total_recall = 0.0;
    for (size_t i = 0; i < queries.size(); ++i) {
        // 获取索引返回的结果，类型是 std::vector<Rec>
        auto index_results = query_futures[i].get();
        // 将结果放入哈希集合中以便快速查找
        std::unordered_set<Rec, de::RecordHash<Rec>> index_result_set(index_results.begin(), index_results.end());

        // --- 暴力计算真值 ---
        const auto& qp_struct = queries[i];
        const Rec& query_point = qp_struct.point;
        const size_t k = qp_struct.k;

        // 使用一个最小堆来找到 top-k，堆顶是距离最远的点
        auto cmp = [&](const Rec& a, const Rec& b) {
            return query_point.calc_distance(a) < query_point.calc_distance(b);
        };
        std::priority_queue<Rec, std::vector<Rec>, decltype(cmp)> top_k_heap(cmp);

        for (const auto& point : ground_truth_data) {
            if (top_k_heap.size() < k) {
                top_k_heap.push(point);
            } else {
                // 如果当前点的距离比堆顶（最远点）的距离还小，则替换
                if (query_point.calc_distance(point) < query_point.calc_distance(top_k_heap.top())) {
                    top_k_heap.pop();
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

    // 4. 计算并打印平均召回率
    // -----------------------------------------------------------------
    if (queries.empty()) {
        fprintf(stdout, "[RESULT] Average Recall: N/A (0 queries)\n");
    } else {
        double average_recall = total_recall / queries.size();
        fprintf(stdout, "[RESULT] Average Recall: %.4f\n", average_recall);
    }

    // =================================================================
    //                     计算 Recall 代码结束
    // =================================================================


  gsl_rng_free(rng);
  delete extension;
  fflush(stderr);
  fflush(stdout);
}