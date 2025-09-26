/*
 * include/shard/HNSW.h
 *
 * Copyright (C) 2024 Your Name <you@example.com>
 * Based on the VPTree implementation by Douglas B. Rumbaugh.
 *
 * Distributed under the Modified BSD License.
 *
 * A shard shim using hnswlib for high-dimensional metric similarity search.
 * This implementation treats the HNSW index as a static structure built
 * at creation time. Updates and deletes are handled by the DynamicExtension
 * framework's buffering and recreation mechanism.
 */
#pragma once

#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <omp.h>

#include "framework/ShardRequirements.h"
#include "hnswlib/hnswlib.h"
#include "psu-ds/PriorityQueue.h"

// Re-using types from the framework's common utilities
using psudb::byte;
using psudb::CACHELINE_SIZE;
using psudb::PriorityQueue;

namespace de {

template <NDRecordInterface R, size_t M = 64, size_t EF_CONSTRUCTION = 256,
          size_t EF_SEARCH = 200>
class HNSW {
public:
    typedef R RECORD;

private:
    Wrapped<R> *m_data;
    size_t m_reccnt;
    size_t m_alloc_size;

    hnswlib::L2SpaceI *m_space;
    hnswlib::HierarchicalNSW<int> *m_hnsw_index;

    std::unordered_map<R, size_t, RecordHash<R>> m_lookup_map;

public:
    HNSW(BufferView<R> buffer)
        : m_data(nullptr), m_reccnt(0), m_space(nullptr), m_hnsw_index(nullptr) {
        
        m_alloc_size = psudb::sf_aligned_alloc(
            CACHELINE_SIZE, buffer.get_record_count() * sizeof(Wrapped<R>), (byte **)&m_data);

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

    HNSW(std::vector<HNSW *> shards)
        : m_reccnt(0), m_space(nullptr), m_hnsw_index(nullptr) {

        size_t total_reccnt = 0;
        for (const auto &shard : shards) {
            total_reccnt += shard->get_record_count();
        }

        m_alloc_size = psudb::sf_aligned_alloc(
            CACHELINE_SIZE, total_reccnt * sizeof(Wrapped<R>), (byte **)&m_data);

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

    ~HNSW() {
        delete m_hnsw_index;
        delete m_space;
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
        m_hnsw_index->setEf(EF_SEARCH);

        auto result_queue = m_hnsw_index->searchKnn(point.data, k);

        // hnswlib's queue is a max-heap (farthest is at the top). We need to
        // extract the results to push into the framework's priority queue.
        size_t result_size = result_queue.size();
        std::vector<hnswlib::labeltype> labels(result_size);
        for (int i = result_size - 1; i >= 0; --i) {
            labels[i] = result_queue.top().second;
            result_queue.pop();
        }

        // Push results into the framework's priority queue.
        // The framework expects pointers to the Wrapped<R> records.
        for (const auto &label : labels) {
            pq.push(&m_data[label]);
        }
    }

    Wrapped<R> *point_lookup(const R &rec, bool filter = false) {
        m_hnsw_index->setEf(10);
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
        if (idx >= m_reccnt) return nullptr;
        return &m_data[idx];
    }

    size_t get_tombstone_count() const {
        return 0;
    }

private:
    void build_index() {
        const size_t dim = 128;
    
        m_space = new hnswlib::L2SpaceI(dim);
        m_hnsw_index = new hnswlib::HierarchicalNSW<int>(m_space, m_reccnt, M,
                                                            EF_CONSTRUCTION);
        // #pragma omp parallel for
        for (size_t i = 0; i < m_reccnt; ++i) {
            m_hnsw_index->addPoint(m_data[i].rec.data, i);
        }
    }
};

} // namespace de