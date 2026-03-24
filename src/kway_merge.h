#pragma once
// K-way merge and two-way merge utilities.
// Shared between parquet_sort (stage 1) and sorted_npy_merge (stage 2).

#include <cstdint>
#include <queue>
#include <vector>

// One pre-sorted sub-sequence for the K-way merge.
struct MergeSequence {
    const int64_t*  times;   // sort key 1: time (ns)
    const int8_t*   types;   // sort key 2: type_priority per element
    const int64_t*  codes;   // sort key 3: stock_code per element
    const uint32_t* locs;    // original row index in source table
    int64_t length;
};

struct HeapEntry {
    int64_t  time;
    int8_t   type;
    int64_t  code;
    uint32_t loc;
    int      seq_idx;
    int64_t  pos;

    bool operator>(const HeapEntry& o) const {
        if (time != o.time) return time > o.time;
        if (type != o.type) return type > o.type;
        return code > o.code;
    }
};

// K-way merge of sorted sequences using a min-heap.
// Output: out_src[i] = type_priority (0=tick,1=order,2=trans),
//         out_loc[i] = row index in the corresponding source table.
inline void kway_merge(
    const std::vector<MergeSequence>& seqs,
    uint8_t*  out_src,
    uint32_t* out_loc,
    int64_t   total)
{
    std::priority_queue<HeapEntry, std::vector<HeapEntry>,
                        std::greater<HeapEntry>> heap;

    for (int i = 0; i < static_cast<int>(seqs.size()); ++i) {
        if (seqs[i].length > 0) {
            heap.push({seqs[i].times[0], seqs[i].types[0],
                        seqs[i].codes[0], seqs[i].locs[0], i, 0});
        }
    }

    int64_t out_idx = 0;
    while (!heap.empty()) {
        auto top = heap.top();
        heap.pop();

        out_src[out_idx] = static_cast<uint8_t>(top.type);
        out_loc[out_idx] = top.loc;
        ++out_idx;

        int64_t next = top.pos + 1;
        int si = top.seq_idx;
        if (next < seqs[si].length) {
            heap.push({seqs[si].times[next], seqs[si].types[next],
                        seqs[si].codes[next], seqs[si].locs[next],
                        si, next});
        }
    }
}

// Two-pointer merge of two arrays sorted by biz_index.
// Returns indices into the conceptual concatenation [a | b]:
//   result[k] < a_len  →  from a at index result[k]
//   result[k] >= a_len →  from b at index (result[k] - a_len)
inline std::vector<uint32_t> two_way_merge_by_biz(
    const int64_t* a_biz, int64_t a_len,
    const int64_t* b_biz, int64_t b_len)
{
    std::vector<uint32_t> result;
    result.reserve(a_len + b_len);
    int64_t i = 0, j = 0;
    while (i < a_len && j < b_len) {
        if (a_biz[i] <= b_biz[j]) {
            result.push_back(static_cast<uint32_t>(i++));
        } else {
            result.push_back(static_cast<uint32_t>(a_len + j));
            ++j;
        }
    }
    while (i < a_len) result.push_back(static_cast<uint32_t>(i++));
    while (j < b_len) { result.push_back(static_cast<uint32_t>(a_len + j)); ++j; }
    return result;
}
