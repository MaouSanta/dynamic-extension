#pragma once

#include <cstdlib>
#include "psu-ds/BTree.h"
#include "framework/interface/Record.h"
#include "pgm/pgm_index_dynamic.hpp"

/* BTree definitions*/
template <typename K, typename V>
struct btree_record {
    K key;
    V value;

   inline bool operator<(const btree_record& other) const {
        return key < other.key || (key == other.key && value < other.value);
    }

    inline bool operator==(const btree_record& other) const {
        return key == other.key && value == other.value;
    }
};

template <typename K, typename V>
struct btree_key_extract {
    static const K &get(const btree_record<K, V> &v) {
        return v.key;
    }
};

typedef psudb::BTree<int64_t, btree_record<int64_t, int64_t>, btree_key_extract<int64_t, int64_t>> BenchBTree;


/*MTree Definitions*/

const size_t W2V_SIZE = 300;
typedef de::EuclidPoint<double, W2V_SIZE> Word2VecRec;

const size_t ANNSize = 128;
// typedef de::EuclidPoint<uint64_t, ANNSize> ANNRec;
typedef de::EuclidPoint<float, ANNSize> ANNRec;

struct euclidean_distance {
    double operator()(const Word2VecRec &first, const Word2VecRec &second) const {
        double dist = 0;
        for (size_t i=0; i<W2V_SIZE; i++) {
            dist += (first.data[i] - second.data[i]) * (first.data[i] - second.data[i]);
        }
        
        return std::sqrt(dist);
    }

    double operator()(const ANNRec &first, const ANNRec &second) const {
        double dist = 0;
        for (size_t i=0; i<ANNSize; i++) {
            dist += ((double) first.data[i] - (double) second.data[i]) * ((double) first.data[i] - (double) second.data[i]);
        }
        
        return std::sqrt(dist);
    }
};

#ifdef _GNU_SOURCE
#include "mtree.h"
typedef mt::mtree<Word2VecRec, euclidean_distance> MTree;
typedef mt::mtree<ANNRec, euclidean_distance> MTree_alt;
#endif

typedef pgm::DynamicPGMIndex<uint64_t, uint64_t, pgm::PGMIndex<uint64_t, 64>> PGM;

