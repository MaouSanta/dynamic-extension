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
#include "psu-ds/PriorityQueue.h"
#include "efanna2e/index_graph.h"
#include "efanna2e/index_random.h"

// Re-using types from the framework's common utilities
using psudb::byte;
using psudb::CACHELINE_SIZE;
using psudb::PriorityQueue;

namespace de {

template <NDRecordInterface R>
class NSG {
public:
    typedef R RECORD;

private:
    Wrapped<R> *m_data;
    size_t m_reccnt;
    size_t m_alloc_size;

    efanna2e::IndexGraph* m_index;

public:
    NSG(BufferView<R> buffer)
        : m_data(nullptr), m_reccnt(0), m_index(nullptr) {
        m_alloc_size = psudb::sf_aligned_alloc(
            CACHELINE_SIZE, buffer.get_record_count() * sizeof(Wrapped<R>), &m_data);
        
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

    NSG(std::vector<NSG *> shards)
        : m_reccnt(0), m_index(nullptr) {

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

    ~NSG() {
        delete m_index;
        if (m_data) {
            free(m_data);
        }
    }

    size_t get_record_count() const { return m_reccnt; }

    size_t get_memory_usage() const { return m_alloc_size; }

    size_t get_aux_memory_usage() const { return 0; }

    void search(const R &point, size_t k,
                PriorityQueue<Wrapped<R>, DistCmpMax<Wrapped<R>>> &pq) {
        if (!m_index || m_reccnt == 0) {
            return;
        }

        std::vector<unsigned> labels(k);
        efanna2e::Parameters paras;
        paras.Set<unsigned>("L_search", 200);
        m_index->Search((float *)point.data, (float *)m_data->rec.data, k, paras, labels.data());

        // Push results into the framework's priority queue.
        // The framework expects pointers to the Wrapped<R> records.
        for (const auto &label : labels) {
            pq.push(&m_data[label]);
        }
    }

    Wrapped<R> *point_lookup(const R &rec, bool filter = false) {
        // m_hnsw_index->setEf(10);
        // auto ret = m_hnsw_index->searchKnn(rec.data, 1);
        // auto i = ret.top().second;
        // if (m_data[i].rec == rec) {
        //     return &m_data[i];
        // }
        
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
        
        unsigned K = 100;
        unsigned L = 100;
        unsigned iter = 1;
        unsigned S = 100; // ef_search
        unsigned R_ = 100;

        efanna2e::IndexRandom init_index(dim, m_reccnt);
        m_index = new efanna2e::IndexGraph(dim, m_reccnt, efanna2e::L2, (efanna2e::Index*)(&init_index));

        efanna2e::Parameters paras;
        paras.Set<unsigned>("K", K);
        paras.Set<unsigned>("L", L);
        paras.Set<unsigned>("iter", iter);
        paras.Set<unsigned>("S", S);
        paras.Set<unsigned>("R", R_);

        m_index->Build(m_reccnt, (float *)m_data->rec.data, paras);
    }
};
} // namespace de
