/*
 * include/shard/DiskANN.h
 *
 * Copyright (C) 2024 Your Name <you@example.com>
 * Based on the HNSW implementation.
 *
 * Distributed under the Modified BSD License.
 *
 * A shard shim using Microsoft's DiskANN for high-dimensional metric similarity search.
 * This implementation uses DiskANN's **in-memory index**. The index is built
 * at creation time and held entirely in RAM. Updates and deletes are handled by
 * the DynamicExtension framework's buffering and recreation mechanism.
 */
#pragma once

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <omp.h>
#include <cstring> // For memcpy

#include "framework/ShardRequirements.h"
#include "psu-ds/PriorityQueue.h"

// DiskANN headers for the IN-MEMORY index
#include "diskann/include/index.h"
#include "diskann/include/parameters.h"
#include "diskann/include/distance.h"

// Re-using types from the framework's common utilities
using psudb::byte;
using psudb::CACHELINE_SIZE;
using psudb::PriorityQueue;

namespace de {

// Template parameters for in-memory DiskANN:
// R_GRAPH: Graph degree (similar to HNSW's M)
// L_BUILD: Build-time search list size (similar to HNSW's efConstruction)
// L_SEARCH: Query-time search list size (similar to HNSW's efSearch)
template <NDRecordInterface Rec, unsigned R_GRAPH = 64, unsigned L_BUILD = 100,
          unsigned L_SEARCH = 100>
class DiskANN {
public:
    typedef Rec RECORD;

private:
    // DiskANN operates on raw data types. Assuming your ANNRec has 'int' data.
    using DataType = float; 

    Wrapped<Rec> *m_data;
    size_t m_reccnt;
    size_t m_alloc_size;

    // Pointer to the in-memory DiskANN index object.
    // The second template parameter 'size_t' is the type for tags (record indices).
    diskann::Index<DataType, size_t> *m_diskann_index;

public:
    DiskANN(BufferView<Rec> buffer)
        : m_data(nullptr), m_reccnt(0), m_diskann_index(nullptr) {
        
        m_alloc_size = psudb::sf_aligned_alloc(
            CACHELINE_SIZE, buffer.get_record_count() * sizeof(Wrapped<Rec>), (byte **)&m_data);

        for (size_t i = 0; i < buffer.get_record_count(); i++) {
            auto rec = buffer.get(i);
            if (rec->is_deleted()) {
                continue;
            }
            m_data[m_reccnt++] = *rec;
        }

        if (m_reccnt > 0) {
            build_index();
        }
    }

    DiskANN(std::vector<DiskANN *> shards)
        : m_reccnt(0), m_diskann_index(nullptr) {

        size_t total_reccnt = 0;
        for (const auto &shard : shards) {
            total_reccnt += shard->get_record_count();
        }

        m_alloc_size = psudb::sf_aligned_alloc(
            CACHELINE_SIZE, total_reccnt * sizeof(Wrapped<Rec>), (byte **)&m_data);

        for (const auto &shard : shards) {
            for (size_t j = 0; j < shard->get_record_count(); j++) {
                const auto *rec = shard->get_record_at(j);
                if (rec->is_deleted()) {
                    continue;
                }
                m_data[m_reccnt++] = *rec;
            }
        }

        if (m_reccnt > 0) {
            build_index();
        }
    }

    ~DiskANN() {
        delete m_diskann_index; // Deletes the in-memory index object
        if (m_data) {
            free(m_data);
        }
    }

    size_t get_record_count() const { return m_reccnt; }
    size_t get_memory_usage() const { return m_alloc_size; }
    size_t get_tombstone_count() const { return 0; }

    size_t get_aux_memory_usage() const {
        if (m_diskann_index) {
            // Estimate memory usage: graph + data copy
            // Graph: num_points * degree * sizeof(neighbor_id)
            // Data: num_points * dim * sizeof(DataType)
            // This is a rough estimation.
            size_t dim = 128;
            size_t graph_mem = m_reccnt * R_GRAPH * sizeof(unsigned);
            size_t data_mem = m_reccnt * dim * sizeof(DataType);
            return graph_mem + data_mem;
        }
        return 0;
    }

    /*************************************************************************
     *                  search (In-Memory Version)
     *************************************************************************/
    void search(const Rec &point, size_t k,
                PriorityQueue<Wrapped<Rec>, DistCmpMax<Wrapped<Rec>>> &pq) {
        if (!m_diskann_index || m_reccnt == 0) return;

        const size_t dim = 128;
        std::vector<float> raw_query(dim);
        for (size_t d = 0; d < dim; ++d)
            raw_query[d] = static_cast<float>(point.data[d]);

        std::vector<size_t> tags(k);
        std::vector<float> dists(k);

        auto [result_count, _] = m_diskann_index->search(raw_query.data(), k, L_SEARCH, tags.data(), dists.data());

        for (size_t i = 0; i < result_count; ++i)
            pq.push(&m_data[tags[i]]);
    }

    /*************************************************************************
     *                  point_lookup (线性查找)
     *************************************************************************/
    Wrapped<Rec> *point_lookup(const Rec &rec, bool filter = false) {
        for (size_t j = 0; j < m_reccnt; ++j) {
            if (m_data[j].rec == rec) {
                return &m_data[j];
            }
        }
        return nullptr;
    }


    const Wrapped<Rec> *get_record_at(size_t idx) const {
        if (idx >= m_reccnt) return nullptr;
        return &m_data[idx];
    }

private:
    void build_index() {
        const size_t dim = 128;
        using DataType = float;

        // Step 1: 将 Wrapped<Rec> 转为连续数组
        std::vector<DataType> raw_data(m_reccnt * dim);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < m_reccnt; ++i)
            for (size_t d = 0; d < dim; ++d)
                raw_data[i * dim + d] = static_cast<DataType>(m_data[i].rec.data[d]);

        // Step 2: 标签数组
        std::vector<size_t> tags(m_reccnt);
        for (size_t i = 0; i < m_reccnt; ++i) tags[i] = i;

        // Step 3: 构建 IndexConfig
        auto write_params = diskann::IndexWriteParametersBuilder(L_BUILD, R_GRAPH)
                                .with_num_threads(omp_get_max_threads())
                                .build();

        auto index_config = diskann::IndexConfigBuilder()
                                .with_metric(diskann::Metric::L2)     // 使用 L2 距离
                                .with_dimension(dim)
                                .with_max_points(m_reccnt)
                                // 移除 DataStoreStrategy 的设置，因为 Index 构造函数将接管内存数据。
                                .with_index_write_params(write_params)
                                .build();

        // **--------------------------------------------------------------**
        // ** 原始代码中的 Step 4, 5, 6, 7 将被替换为以下简洁版本：**
        // **--------------------------------------------------------------**
        
        // Step 4: 创建 Index 对象。
        // 对于 in-memory 索引，通常只需要 IndexConfig。
        // Index 将在内部创建所需的数据存储和图存储。
        m_diskann_index = new diskann::Index<DataType, size_t>(
            index_config,
            nullptr, // data_store (设置为 nullptr，让 Index 内部创建)
            nullptr, // graph_store
            nullptr  // tag_store
        );

        // Step 5: 构建索引
        // DiskANN 的 build() 函数通常接受原始数据指针、点数量和标签。
        m_diskann_index->build(raw_data.data(), m_reccnt, tags);
    }

};

} // namespace de