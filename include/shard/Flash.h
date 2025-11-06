/*
 * include/shard/Flash.h
 *
 * Copyright (C) 2024 Your Name <you@example.com>
 * Based on the VPTree implementation by Douglas B. Rumbaugh.
 *
 * Distributed under the Modified BSD License.
 *
 * A shard shim using hnswlib for high-dimensional metric similarity search.
 * This implementation treats the Flash index as a static structure built
 * at creation time. Updates and deletes are handled by the DynamicExtension
 * framework's buffering and recreation mechanism.
 */
#pragma once

#include "Eigen/Dense"
#include <cfloat>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "framework/ShardRequirements.h"
#include "hnswlib/hnswalg_flash.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_flash.h"
#include "psu-ds/PriorityQueue.h"

// Re-using types from the framework's common utilities
using psudb::byte;
using psudb::CACHELINE_SIZE;
using psudb::PriorityQueue;

using Eigen::MatrixXf;
using Eigen::VectorXf;

namespace de {

template <NDRecordInterface R, size_t M = 32, size_t EF_CONSTRUCTION = 1024,
          size_t EF_SEARCH = 100>
class Flash {
public:
  typedef R RECORD;

private:
  Wrapped<R> *m_data;
  size_t m_reccnt;
  size_t m_alloc_size;

  hnswlib::FlashSpace *m_space;
  hnswlib::HierarchicalNSWFlash<float> *m_hnsw_index;
  float *codebooks;

  const size_t subvector_num_ = 32;
  const size_t cluster_num_ = 16;
  size_t sample_num;

  size_t data_dim_;
  const size_t byte_num_ = subvector_num_ / 2;
  const size_t ori_dim_ = 128;      // The original dim of data before PCA
  const size_t principal_dim_ = 96; // The dimension of the principal components
  float qmin, qmax;                 // The min and max bounds of SQ

  Eigen::VectorXf data_mean_;           // Mean of data
  Eigen::MatrixXf principal_components; // Principal components

  std::unordered_map<R, size_t, RecordHash<R>> m_lookup_map;

public:
  Flash(BufferView<R> buffer)
      : m_data(nullptr), m_reccnt(0), m_alloc_size(0), m_hnsw_index(nullptr) {

    m_alloc_size = psudb::sf_aligned_alloc(
        CACHELINE_SIZE, buffer.get_record_count() * sizeof(Wrapped<R>),
        &m_data);

    for (size_t i = 0; i < buffer.get_record_count(); i++) {
      auto rec = buffer.get(i);
      if (rec->is_deleted()) {
        continue;
      }

      m_data[m_reccnt] = *rec;
      m_reccnt++;
    }

    if (m_reccnt > 0) {
      build_index();
    }
  }

  Flash(std::vector<Flash *> shards)
      : m_reccnt(0), m_alloc_size(0), m_hnsw_index(nullptr) {

    size_t total_reccnt = 0;
    for (const auto &shard : shards) {
      total_reccnt += shard->get_record_count();
    }

    m_alloc_size = psudb::sf_aligned_alloc(
        CACHELINE_SIZE, total_reccnt * sizeof(Wrapped<R>), &m_data);

    for (const auto &shard : shards) {
      for (size_t j = 0; j < shard->get_record_count(); j++) {
        const auto *rec = shard->get_record_at(j);
        if (rec->is_deleted()) {
          continue;
        }

        m_data[m_reccnt] = *rec;
        m_reccnt++;
      }
    }

    if (m_reccnt > 0) {
      build_index();
    }
  }

  ~Flash() {
    if (codebooks) {
      free(codebooks);
    }
    if (m_space) {
      delete m_space;
    }
    if (m_hnsw_index) {
      delete m_hnsw_index;
    }
    if (m_data) {
      free(m_data);
    }
  }

  size_t get_record_count() const { return m_reccnt; }

  size_t get_memory_usage() const { return m_alloc_size; }

  size_t get_aux_memory_usage() const {
    if (m_hnsw_index) {
      return m_hnsw_index->max_elements_ * m_hnsw_index->size_data_per_element_;
    }
    return 0;
  }

  void search(const R &point, size_t k,
              PriorityQueue<Wrapped<R>, DistCmpMax<Wrapped<R>>> &pq) {
    if (!m_hnsw_index || m_reccnt == 0) {
      return;
    }

    Eigen::VectorXf query =
        Eigen::Map<const Eigen::VectorXf>(point.data, ori_dim_);
    pcaEncode(query);

    std::vector<uint8_t> encoded_query(subvector_num_ * cluster_num_ +
                                       byte_num_);

    pqEncode(query, encoded_query.data() + subvector_num_ * cluster_num_,
             encoded_query.data());

    // 搜索 k * 10000 个点，实际上返回的数量真不一定
    std::priority_queue<std::pair<float, hnswlib::labeltype>> tmp =
        m_hnsw_index->searchKnn(encoded_query.data(), k);

    // std::cout << m_reccnt << " " << tmp.size() << std::endl;
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
    while (!tmp.empty()) {
      float res = 0;
      size_t a = tmp.top().second;
      for (int j = 0; j < ori_dim_; ++j) {
        float t = m_data[a].rec.data[j] - point.data[j];
        res += t * t;
      }
      result.emplace(res, a);
      if (result.size() > k) {
        result.pop();
      }
      tmp.pop();
    }

    while (!result.empty()) {
      pq.push(&m_data[result.top().second]);
      result.pop();
    }
  }

  Wrapped<R> *point_lookup(const R &rec, bool filter = false) {
    m_hnsw_index->setEf(100);
    auto ret = m_hnsw_index->searchKnn(rec.data, 1);
    auto i = ret.top().second;
    if (m_data[i].rec == rec) {
      return &m_data[i];
    }

    for (int j = 0; j < m_reccnt; ++j) {
      if (m_data[j].rec == rec) {
        return &m_data[j];
      }
    }
    return nullptr;
  }

  Wrapped<R> *get_data() const { return m_data; }

  const Wrapped<R> *get_record_at(size_t idx) const {
    if (idx >= m_reccnt)
      return nullptr;
    return &m_data[idx];
  }

  size_t get_tombstone_count() const { return 0; }

private:
  void build_index() {
    qmin = qmax = 0;
    sample_num = std::min(m_reccnt, (size_t)10000);

    data_dim_ = ori_dim_;

    m_space = new hnswlib::FlashSpace(byte_num_);
    m_hnsw_index = new hnswlib::HierarchicalNSWFlash<float>(m_space, m_reccnt,
                                                            M, EF_CONSTRUCTION);
    m_hnsw_index->setEf(EF_SEARCH);

    m_hnsw_index->flash_data_dist_table_ =
        (uint8_t *)malloc(m_reccnt * subvector_num_ * cluster_num_);

    // 数据类型转换
    Eigen::MatrixXf data_set(m_reccnt, ori_dim_);
#pragma omp parallel for
    for (int i = 0; i < m_reccnt; ++i) {
      Eigen::Map<Eigen::VectorXf> vec(m_data[i].rec.data, ori_dim_);
      data_set.row(i) = vec.transpose();
    }

    // PCA
    generate_matrix(data_set);
    pcaEncode(data_set);
    data_dim_ = principal_dim_;

    // PQ codebooks
    generate_codebooks(data_set);
#pragma omp parallel for
    for (size_t i = 0; i < m_reccnt; ++i) {
      std::vector<uint8_t> encoded_data(subvector_num_ * cluster_num_ +
                                        byte_num_);
      pqEncode(data_set.row(i),
               encoded_data.data() + subvector_num_ * cluster_num_,
               encoded_data.data());
      m_hnsw_index->addPoint(encoded_data.data(), i);
    }
  }

  void generate_codebooks(Eigen::MatrixXf &data_set) {
    size_t data_num = data_set.rows();
    int sub_dim = data_dim_ / subvector_num_;

    std::vector<size_t> indices(data_num);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(19260817);

    std::shuffle(indices.begin(), indices.end(), g);

    // 分配 codebook
    codebooks = (float *)malloc(cluster_num_ * subvector_num_ * sub_dim *
                                sizeof(float));

#pragma omp parallel for
    for (size_t i = 0; i < subvector_num_; ++i) {
      Eigen::MatrixXf subvector_data(sample_num, sub_dim);

      for (size_t j = 0; j < sample_num; ++j) {
        const size_t row_idx = indices[j];
        const float *ptr = data_set.row(row_idx).data() + i * sub_dim;
        subvector_data.row(j) =
            Eigen::Map<const Eigen::VectorXf>(ptr, sub_dim).transpose();
      }

      // KMeans 聚类
      Eigen::MatrixXf centroid_matrix =
          kMeans(subvector_data, cluster_num_, 12);

      // 写入 codebooks
      for (int r = 0; r < centroid_matrix.rows(); ++r) {
        Eigen::VectorXf row = centroid_matrix.row(r);
        std::copy(row.data(), row.data() + row.size(),
                  codebooks + r * data_dim_ + i * sub_dim);
      }
    }

    for (size_t i = 0; i < subvector_num_; ++i) {
      for (size_t j = 0; j < cluster_num_; ++j) {
        for (size_t k = 0; k < cluster_num_; ++k) {
          float dist = 0;
          for (size_t l = 0; l < sub_dim; ++l) {
            dist += (codebooks[j * data_dim_ + i * sub_dim + l] -
                     codebooks[k * data_dim_ + i * sub_dim + l]) *
                    (codebooks[j * data_dim_ + i * sub_dim + l] -
                     codebooks[k * data_dim_ + i * sub_dim + l]);
          }
          dist = sqrt(dist);
          qmax = std::max(qmax, dist);
          qmin = std::min(qmin, dist);
        }
      }
    }
  }

  /**
   * Perform k-means clustering on the given dataset
   * @param data_set Pointer to the dataset
   * @param cluster_num Number of clusters
   * @param max_iterations Maximum number of iterations
   * @return Returns the cluster center matrix
   */
  MatrixXf kMeans(const MatrixXf &data_set, size_t cluster_num,
                  size_t max_iterations) {
    // Initialize centroids randomly
    size_t data_num = data_set.rows();
    size_t data_dim = data_set.cols();
    MatrixXf centroids(cluster_num, data_dim);
    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(114514);
    std::uniform_int_distribution<> dis(0, data_num - 1);
    for (size_t i = 0; i < cluster_num; ++i) {
      centroids.row(i) = data_set.row(dis(gen));
    }

    // kMeans
    std::vector<size_t> labels(data_num);
    auto startTime = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < max_iterations; ++iter) {

      // Assign labels to each data point, that is, find the nearest cluster
      // center to it.
      for (size_t i = 0; i < data_num; ++i) {
        float min_dist = FLT_MAX;
        size_t best_index = 0;
        for (size_t j = 0; j < cluster_num; ++j) {
          float dist = (data_set.row(i) - centroids.row(j)).squaredNorm();
          if (dist < min_dist) {
            min_dist = dist;
            best_index = j;
          }
        }
        labels[i] = best_index;
      }

      // Update the cluster centers, calculating the mean of all data points
      // in each cluster as the new cluster center.
      MatrixXf new_centroids = MatrixXf::Zero(cluster_num, data_dim);
      std::vector<int> counts(cluster_num, 0);
      for (size_t i = 0; i < data_num; ++i) {
        new_centroids.row(labels[i]) += data_set.row(i);
        counts[labels[i]]++;
      }
      for (size_t j = 0; j < cluster_num; ++j) {
        if (counts[j] != 0) {
          new_centroids.row(j) /= counts[j];
        } else {
          new_centroids.row(j) =
              data_set.row(dis(gen)); // Reinitialize a random centroid if no
                                      // points are assigned
        }
      }
      centroids = new_centroids;
    }
    return centroids;
  }

  /**
   * Perform PQ encoding on the given data and compute the distance table
   * between the encoded vectors and the original data. Then, perform SQ
   * encoding on the distance table with an upper bound of the sum of the
   * maximum distance from each subvector. When encoding base data, the
   * distance table with qmin and qmax remains stable. When encoding query
   * data, the distance table with qmin and qmax needs to be recalculated.
   * @param data Pointer to the data to be encoded
   * @param encoded_vector Pointer to the encoded vector
   * @param dist_table Pointer to the distance table
   * @param is_query Flag indicating whether the data is a query: 1 for query
   * data, 0 for non-query data
   */
  void pqEncode(const Eigen::VectorXf &data, uint8_t *encoded_vector,
                uint8_t *dist_table) {
    size_t dim = data.size();
    size_t sub_dim = dim / subvector_num_;

    const float alpha = 255.0f / (qmax - qmin);

    // Compute distances
    for (size_t i = 0; i < subvector_num_; ++i) {
      uint8_t best_index = 0;
      float min_dist = FLT_MAX;
      for (size_t j = 0; j < cluster_num_; ++j) {
        float res = 0;
        for (size_t k = 0; k < sub_dim; ++k) {
          float t =
              data(i * sub_dim + k) - codebooks[j * dim + i * sub_dim + k];
          res += t * t;
        }
        res = sqrt(res);
        if (res < min_dist) {
          min_dist = res;
          best_index = j;
        }
        float value = (res - qmin) * alpha;
        int quantized_value =
            (int)std::round(std::max(0.0f, std::min(255.0f, value)));
        dist_table[i * cluster_num_ + j] = (uint8_t)quantized_value;
      }
      // 偶数高位 奇数低位
      if (i % 2 == 0) {
        encoded_vector[i / 2] =
            (encoded_vector[i / 2] & 0x0F) | (best_index << 4);
      } else {
        encoded_vector[i / 2] = (encoded_vector[i / 2] & 0xF0) | best_index;
      }
    }
  }

  void generate_matrix(Eigen::MatrixXf &data_set) {
    size_t data_num = data_set.rows();
    size_t data_dim = data_set.cols();

    if (sample_num > data_num) {
      sample_num = data_num;
    }

    std::vector<size_t> indices(data_num);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(19260817);
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::MatrixXf data(sample_num, data_dim);

    // 随机采样 sample_num 行进入 data
    for (int i = 0; i < sample_num; ++i) {
      size_t idx = indices[i];
      data.row(i) = data_set.row(idx);
    }

    // 计算均值 (1 x D)
    data_mean_ = data.colwise().mean();

    // 中心化
    for (int i = 0; i < sample_num; ++i) {
      data.row(i) -= data_mean_.transpose();
    }

    // 协方差矩阵 = (D x N) * (N x D) / (N-1) = (D x D)
    Eigen::MatrixXf covariance_matrix =
        (data.adjoint() * data) / float(sample_num - 1);

    // 特征分解 (协方差矩阵对称 → 自伴随求解)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(
        covariance_matrix);
    principal_components = eigensolver.eigenvectors();

    // Eigen 默认特征值升序 → 我们反转矩阵列顺序
    principal_components = principal_components.rowwise().reverse();

    // 保留前 PRINCIPAL_DIM 列
    principal_components.conservativeResize(Eigen::NoChange, principal_dim_);
  }

  void pcaEncode(Eigen::MatrixXf &data) {
    size_t data_num = data.rows();
    size_t data_dim = data.cols();

    for (size_t i = 0; i < data_num; ++i) {
      data.row(i) -= data_mean_.transpose();
    }

    data = data * principal_components;
  }

  void pcaEncode(Eigen::VectorXf &data) {
    data -= data_mean_;
    data = principal_components.transpose() * data;
  }
};

} // namespace de