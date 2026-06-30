// parquet_sort.cpp  –  C++ port of parquet_sort_modified.py + sorted_npy_merge.py
//
// Stage 1: K-way merge sort for tick / order / trans parquet quote data.
// Stage 2: Join with NPY local times, re-merge front half, per-type output.
//
// Sorting rules (identical to Python version):
//   1. time ascending  (exchange_time in nanoseconds)
//   2. type priority   (tick=0 < order=1 < trans=2)
//   3. stock code asc
//   For SH channels: biz_index ordering takes priority within each channel.
//   For SZ channels: ApplSeq (order_index/trade_index) ordering within each channel.
//
// Build:
//   mkdir build && cd build && cmake .. && make -j

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/util/bitmap_ops.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <numeric>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kway_merge.h"
#include "npy_reader.h"

namespace fs = std::filesystem;

// ── Constants ────────────────────────────────────────────────────────

static constexpr int64_t CHUNK_SIZE           = 20'000'000;
static constexpr int32_t SH_CHANNEL_THRESHOLD = 1000;

// ── Logging ──────────────────────────────────────────────────────────

static FILE* g_log_fp = nullptr;

static double now_sec() {
    using clk = std::chrono::steady_clock;
    static auto t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

#define LOG(fmt, ...) do { \
    auto _now = std::chrono::system_clock::now();                     \
    auto _tt  = std::chrono::system_clock::to_time_t(_now);           \
    struct tm _tm; localtime_r(&_tt, &_tm);                           \
    char _tb[32]; strftime(_tb, sizeof(_tb), "%H:%M:%S", &_tm);      \
    fprintf(stderr, "[%s] " fmt "\n", _tb, ##__VA_ARGS__);            \
    fflush(stderr);                                                   \
    if (g_log_fp) {                                                   \
        fprintf(g_log_fp, "[%s] " fmt "\n", _tb, ##__VA_ARGS__);     \
        fflush(g_log_fp);                                             \
    }                                                                 \
} while (0)

// ── Arrow helpers ────────────────────────────────────────────────────

#define ARROW_OK(expr) do {                                        \
    auto _s = (expr);                                              \
    if (!_s.ok()) {                                                \
        fprintf(stderr, "ARROW ERROR %s:%d: %s\n",                \
                __FILE__, __LINE__, _s.ToString().c_str());        \
        std::exit(1);                                              \
    }                                                              \
} while (0)

template <typename T>
static T unwrap(arrow::Result<T> r, const char* ctx = "") {
    if (!r.ok()) {
        fprintf(stderr, "ARROW ERROR (%s): %s\n", ctx, r.status().ToString().c_str());
        std::exit(1);
    }
    return std::move(r).ValueOrDie();
}

static std::shared_ptr<arrow::Array> combine_chunks(
    const std::shared_ptr<arrow::ChunkedArray>& col)
{
    if (col->num_chunks() == 1) return col->chunk(0);
    return unwrap(arrow::Concatenate(col->chunks(), arrow::default_memory_pool()),
                  "combine_chunks");
}

static std::shared_ptr<arrow::Array> get_column(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& name,
    std::shared_ptr<arrow::DataType> cast_to = nullptr)
{
    auto col = table->GetColumnByName(name);
    if (!col) {
        fprintf(stderr, "Column '%s' not found\n", name.c_str());
        std::exit(1);
    }
    auto arr = combine_chunks(col);
    if (cast_to && !arr->type()->Equals(cast_to)) {
        auto src_id = arr->type()->id();
        if (src_id == arrow::Type::BINARY || src_id == arrow::Type::LARGE_BINARY) {
            auto str_type = (src_id == arrow::Type::LARGE_BINARY)
                                ? arrow::large_utf8() : arrow::utf8();
            arrow::compute::CastOptions s_opts;
            s_opts.to_type = str_type;
            arr = unwrap(arrow::compute::CallFunction("cast", {arr}, &s_opts),
                         "cast_to_str").make_array();
        }
        arrow::compute::CastOptions opts;
        opts.to_type = cast_to;
        arr = unwrap(arrow::compute::CallFunction("cast", {arr}, &opts), "cast")
                  .make_array();
    }
    return arr;
}

static const int64_t* raw_int64(const std::shared_ptr<arrow::Array>& a) {
    return std::static_pointer_cast<arrow::Int64Array>(a)->raw_values();
}
static const int32_t* raw_int32(const std::shared_ptr<arrow::Array>& a) {
    return std::static_pointer_cast<arrow::Int32Array>(a)->raw_values();
}

static std::shared_ptr<arrow::Array> make_const_int8(int64_t n, int8_t v) {
    arrow::Int8Builder b;
    ARROW_OK(b.Reserve(n));
    for (int64_t i = 0; i < n; ++i) b.UnsafeAppend(v);
    return unwrap(b.Finish(), "const_int8");
}

static std::shared_ptr<arrow::Array> make_const_string(int64_t n, const std::string& v) {
    arrow::StringBuilder b;
    ARROW_OK(b.Reserve(n));
    ARROW_OK(b.ReserveData(n * static_cast<int64_t>(v.size())));
    for (int64_t i = 0; i < n; ++i) b.UnsafeAppend(v);
    return unwrap(b.Finish(), "const_string");
}

static std::shared_ptr<arrow::Array> wrap_u32(const uint32_t* data, int64_t n) {
    auto buf = unwrap(arrow::AllocateBuffer(n * sizeof(uint32_t)));
    std::memcpy(buf->mutable_data(), data, n * sizeof(uint32_t));
    return std::make_shared<arrow::UInt32Array>(n, std::move(buf));
}

static std::shared_ptr<arrow::Array> wrap_i64(const int64_t* data, int64_t n) {
    auto buf = unwrap(arrow::AllocateBuffer(n * sizeof(int64_t)));
    std::memcpy(buf->mutable_data(), data, n * sizeof(int64_t));
    return std::make_shared<arrow::Int64Array>(n, std::move(buf));
}

static std::shared_ptr<arrow::Array> wrap_u64(const int64_t* data, int64_t n) {
    auto buf = unwrap(arrow::AllocateBuffer(n * sizeof(uint64_t)));
    auto* dst = reinterpret_cast<uint64_t*>(buf->mutable_data());
    for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<uint64_t>(data[i]);
    return std::make_shared<arrow::UInt64Array>(n, std::move(buf));
}

// ── Target output schemas per quote_struct.h ─────────────────────────

static auto fsl_u32_10 = arrow::fixed_size_list(arrow::field("item", arrow::uint32()), 10);

static std::shared_ptr<arrow::Schema> make_tick_schema() {
    return arrow::schema({
        arrow::field("serial",           arrow::int32()),
        arrow::field("mi_type",          arrow::int32()),
        arrow::field("local_time",       arrow::uint64()),
        arrow::field("exchange_time",    arrow::uint64()),
        arrow::field("wind_code",        arrow::utf8()),
        arrow::field("ticker",           arrow::utf8()),
        arrow::field("action_day",       arrow::int32()),
        arrow::field("trading_day",      arrow::int32()),
        arrow::field("exch_time",        arrow::int32()),
        arrow::field("status",           arrow::int32()),
        arrow::field("pre_close_px",     arrow::uint32()),
        arrow::field("open_px",          arrow::uint32()),
        arrow::field("high_px",          arrow::uint32()),
        arrow::field("low_px",           arrow::uint32()),
        arrow::field("last_px",          arrow::uint32()),
        arrow::field("ap_array",         fsl_u32_10),
        arrow::field("av_array",         fsl_u32_10),
        arrow::field("bp_array",         fsl_u32_10),
        arrow::field("bv_array",         fsl_u32_10),
        arrow::field("num_of_trades",    arrow::uint32()),
        arrow::field("total_vol",        arrow::int64()),
        arrow::field("total_notional",   arrow::int64()),
        arrow::field("total_bid_vol",    arrow::int64()),
        arrow::field("total_ask_vol",    arrow::int64()),
        arrow::field("weighted_avg_bp",  arrow::uint32()),
        arrow::field("weighted_avg_ap",  arrow::uint32()),
        arrow::field("IOPV",             arrow::int32()),
        arrow::field("yield_to_maturity",arrow::int32()),
        arrow::field("upper_limit_px",   arrow::uint32()),
        arrow::field("lower_limit_px",   arrow::uint32()),
        arrow::field("prefix",           arrow::utf8()),
        arrow::field("PE1",              arrow::int32()),
        arrow::field("PE2",              arrow::int32()),
        arrow::field("change",           arrow::int32()),
    });
}

static std::shared_ptr<arrow::Schema> make_trans_schema() {
    return arrow::schema({
        arrow::field("serial",           arrow::int32()),
        arrow::field("mi_type",          arrow::int32()),
        arrow::field("local_time",       arrow::uint64()),
        arrow::field("exchange_time",    arrow::uint64()),
        arrow::field("channel",          arrow::uint16()),
        arrow::field("symbol",           arrow::utf8()),
        arrow::field("market",           arrow::uint8()),
        arrow::field("int_time",         arrow::int32()),
        arrow::field("trade_price",      arrow::int32()),
        arrow::field("bsflag",           arrow::utf8()),
        arrow::field("trade_type",       arrow::utf8()),
        arrow::field("trade_index",      arrow::int64()),
        arrow::field("trade_volume",     arrow::int64()),
        arrow::field("trade_amount",     arrow::int64()),
        arrow::field("sell_id",          arrow::int64()),
        arrow::field("buy_id",           arrow::int64()),
        arrow::field("biz_index",        arrow::int64()),
    });
}

static std::shared_ptr<arrow::Schema> make_order_schema() {
    return arrow::schema({
        arrow::field("serial",           arrow::int32()),
        arrow::field("mi_type",          arrow::int32()),
        arrow::field("local_time",       arrow::uint64()),
        arrow::field("exchange_time",    arrow::uint64()),
        arrow::field("order_type",       arrow::utf8()),
        arrow::field("market",           arrow::uint8()),
        arrow::field("bsflag",           arrow::utf8()),
        arrow::field("symbol",           arrow::utf8()),
        arrow::field("int_time",         arrow::int32()),
        arrow::field("order_index",      arrow::int64()),
        arrow::field("order_volume",     arrow::int64()),
        arrow::field("orderorino",       arrow::uint64()),
        arrow::field("biz_index",        arrow::uint64()),
        arrow::field("order_price",      arrow::int32()),
        arrow::field("channel",          arrow::uint16()),
    });
}

static std::shared_ptr<arrow::Table> promote_to_large(
    std::shared_ptr<arrow::Table> table)
{
    for (int i = 0; i < table->num_columns(); ++i) {
        auto fld = table->schema()->field(i);
        auto tid = fld->type()->id();

        std::shared_ptr<arrow::DataType> new_type;
        if (tid == arrow::Type::BINARY)
            new_type = arrow::large_binary();
        else if (tid == arrow::Type::STRING)
            new_type = arrow::large_utf8();
        else if (tid == arrow::Type::LIST) {
            auto lt = std::static_pointer_cast<arrow::ListType>(fld->type());
            new_type = arrow::large_list(lt->value_type());
        }
        else
            continue;

        arrow::compute::CastOptions opts;
        opts.to_type = new_type;
        auto col = combine_chunks(table->column(i));
        auto casted = unwrap(
            arrow::compute::CallFunction("cast", {col}, &opts),
            "promote_large").make_array();
        auto chunked = std::make_shared<arrow::ChunkedArray>(
            arrow::ArrayVector{casted});
        table = unwrap(table->SetColumn(i, fld->WithType(new_type), chunked),
                       "set_col_promote");
    }
    return table;
}

// Extract a field from a StructArray, cast to int64.
static std::shared_ptr<arrow::Int64Array> struct_field_i64(
    const std::shared_ptr<arrow::StructArray>& sa, const std::string& name)
{
    auto field = sa->GetFieldByName(name);
    if (!field) {
        fprintf(stderr, "struct field '%s' not found\n", name.c_str());
        std::exit(1);
    }
    if (field->type_id() == arrow::Type::INT64)
        return std::static_pointer_cast<arrow::Int64Array>(field);
    arrow::compute::CastOptions opts;
    opts.to_type = arrow::int64();
    return std::static_pointer_cast<arrow::Int64Array>(
        unwrap(arrow::compute::CallFunction("cast", {field}, &opts),
               "struct_field_cast").make_array());
}

// ── Step 1 : Read parquet files ──────────────────────────────────────

static constexpr int READ_THREADS_TOTAL = 16;

struct ReadTask {
    std::string path;
    int type_id;  // 0=tick, 1=trans, 2=order
};

static void read_all_types(
    const std::string& data_dir,
    std::shared_ptr<arrow::Table>& tick,
    std::shared_ptr<arrow::Table>& trans,
    std::shared_ptr<arrow::Table>& order)
{
    auto t0 = now_sec();
    const char* sub_dirs[] = {"1", "2", "3"};
    const char* labels[]   = {"tick ", "trans", "order"};

    // Collect all files across three types
    std::vector<ReadTask> tasks;
    int type_file_counts[3] = {};
    for (int tid = 0; tid < 3; ++tid) {
        std::string folder = data_dir + "/" + sub_dirs[tid];
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(folder)) {
            if (e.path().extension() == ".parquet" &&
                e.path().filename() != "all.parquet")
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
        type_file_counts[tid] = static_cast<int>(files.size());
        for (auto& f : files)
            tasks.push_back({std::move(f), tid});
    }

    int n_tasks = static_cast<int>(tasks.size());
    int n_threads = std::min(READ_THREADS_TOTAL, n_tasks);

    // Thread-local storage: each thread collects tables per type
    struct PerThread { std::vector<std::shared_ptr<arrow::Table>> by_type[3]; };
    std::vector<PerThread> per_thread(n_threads);

    std::atomic<int> next_task{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&tasks, &per_thread, &next_task, n_tasks, t]() {
            auto pool = arrow::default_memory_pool();
            for (;;) {
                int idx = next_task.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n_tasks) break;
                auto& task = tasks[idx];
                auto infile = unwrap(arrow::io::ReadableFile::Open(task.path), "open");
                std::unique_ptr<parquet::arrow::FileReader> reader;
                ARROW_OK(parquet::arrow::OpenFile(infile, pool, &reader));
                std::shared_ptr<arrow::Table> tbl;
                ARROW_OK(reader->ReadTable(&tbl));
                per_thread[t].by_type[task.type_id].push_back(
                    promote_to_large(std::move(tbl)));
            }
        });
    }
    for (auto& th : threads) th.join();

    // Merge per-type tables
    auto pool = arrow::default_memory_pool();
    std::shared_ptr<arrow::Table>* outputs[] = {&tick, &trans, &order};
    for (int tid = 0; tid < 3; ++tid) {
        std::vector<std::shared_ptr<arrow::Table>> tables;
        tables.reserve(type_file_counts[tid]);
        for (auto& pt : per_thread)
            tables.insert(tables.end(),
                          std::make_move_iterator(pt.by_type[tid].begin()),
                          std::make_move_iterator(pt.by_type[tid].end()));
        auto combined = unwrap(arrow::ConcatenateTables(tables, {}, pool), "concat");
        *outputs[tid] = unwrap(combined->CombineChunks(pool), "combine");
        LOG("[%s] %ld rows from %d files", labels[tid],
            (*outputs[tid])->num_rows(), type_file_counts[tid]);
    }
    LOG("All reads done (%.1fs, %d threads)", now_sec() - t0, n_threads);
}

// ── Step 2 : Sort table ──────────────────────────────────────────────

static std::shared_ptr<arrow::Table> sort_table(
    std::shared_ptr<arrow::Table> table,
    const std::vector<std::string>& keys)
{
    arrow::compute::SortOptions opts;
    for (auto& k : keys)
        opts.sort_keys.emplace_back(k, arrow::compute::SortOrder::Ascending);

    auto indices = unwrap(
        arrow::compute::CallFunction("sort_indices", {table}, &opts), "sort_indices");
    return unwrap(
        arrow::compute::CallFunction("take", {table, indices}), "sort_take").table();
}

// ── Step 3 : fix_array_cols (FixedSizeList → List) ───────────────────

static std::shared_ptr<arrow::Table> fix_array_cols(
    std::shared_ptr<arrow::Table> table)
{
    for (int i = 0; i < table->num_columns(); ++i) {
        auto fld = table->schema()->field(i);
        if (fld->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
            auto fsl = std::static_pointer_cast<arrow::FixedSizeListType>(fld->type());
            auto list_type = arrow::list(fsl->value_type());
            arrow::compute::CastOptions opts;
            opts.to_type = list_type;
            auto casted = unwrap(
                arrow::compute::CallFunction("cast", {combine_chunks(table->column(i))}, &opts),
                "fix_fsl").make_array();
            auto new_field = fld->WithType(list_type);
            auto chunked = std::make_shared<arrow::ChunkedArray>(
                arrow::ArrayVector{casted});
            table = unwrap(table->SetColumn(i, new_field, chunked), "set_col_fsl");
        }
    }
    return table;
}

// ── Step 4 : build_meta_with_struct ──────────────────────────────────

static std::shared_ptr<arrow::Table> build_meta_with_struct(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& time_col,
    const std::string& code_col,
    int8_t type_pri,
    const std::string& dtype_str,
    const std::string& struct_name)
{
    int64_t n = table->num_rows();

    auto time_arr = get_column(table, time_col, arrow::int64());
    auto scalar_1000 = arrow::MakeScalar(int64_t(1000));
    auto sort_time = unwrap(
        arrow::compute::CallFunction("multiply", {time_arr, scalar_1000}),
        "multiply").make_array();

    auto stock_code = get_column(table, code_col, arrow::int64());
    auto type_priority = make_const_int8(n, type_pri);
    auto data_type = make_const_string(n, dtype_str);

    arrow::ArrayVector struct_arrays;
    arrow::FieldVector struct_fields;
    for (int i = 0; i < table->num_columns(); ++i) {
        struct_fields.push_back(table->schema()->field(i));
        struct_arrays.push_back(combine_chunks(table->column(i)));
    }
    auto struct_type = arrow::struct_(struct_fields);
    auto struct_arr = unwrap(
        arrow::StructArray::Make(struct_arrays, struct_fields), "make_struct");

    auto schema = arrow::schema({
        arrow::field("sort_time",      arrow::int64()),
        arrow::field("stock_code",     arrow::int64()),
        arrow::field("type_priority",  arrow::int8()),
        arrow::field("data_type",      arrow::utf8()),
        arrow::field(struct_name,      struct_type),
    });

    auto make_chunked = [](std::shared_ptr<arrow::Array> a) {
        return std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{std::move(a)});
    };

    return arrow::Table::Make(schema, {
        make_chunked(sort_time),
        make_chunked(stock_code),
        make_chunked(type_priority),
        make_chunked(data_type),
        make_chunked(std::move(struct_arr)),
    });
}

// ── build_unified_schema ─────────────────────────────────────────────

static std::shared_ptr<arrow::Schema> build_unified_schema(
    const std::shared_ptr<arrow::Schema>& tick_s,
    const std::shared_ptr<arrow::Schema>& order_s,
    const std::shared_ptr<arrow::Schema>& trans_s)
{
    arrow::FieldVector fields;
    for (auto& name : {"sort_time", "type_priority", "stock_code", "data_type"})
        fields.push_back(tick_s->GetFieldByName(name));
    fields.push_back(arrow::field("local_time", arrow::int64()));
    fields.push_back(tick_s->GetFieldByName("tick_data"));
    fields.push_back(order_s->GetFieldByName("order_data"));
    fields.push_back(trans_s->GetFieldByName("trans_data"));
    return arrow::schema(fields);
}

// ── build_sorted_batch (scatter metadata + per-source struct Take) ──

static std::shared_ptr<arrow::UInt32Array> make_scatter_indices(
    const std::vector<uint32_t>& positions, int64_t total_len, int64_t n_valid)
{
    auto idx_buf = unwrap(arrow::AllocateBuffer(total_len * sizeof(uint32_t)));
    auto null_buf = unwrap(arrow::AllocateBitmap(total_len));
    std::memset(null_buf->mutable_data(), 0,
                arrow::bit_util::BytesForBits(total_len));
    auto* idx = reinterpret_cast<uint32_t*>(idx_buf->mutable_data());
    for (uint32_t k = 0; k < static_cast<uint32_t>(n_valid); ++k) {
        idx[positions[k]] = k;
        arrow::bit_util::SetBit(null_buf->mutable_data(), positions[k]);
    }
    return std::make_shared<arrow::UInt32Array>(
        total_len, std::move(idx_buf), std::move(null_buf),
        total_len - n_valid);
}

static std::shared_ptr<arrow::Table> build_sorted_batch(
    const uint8_t* chunk_src, int64_t chunk_len,
    std::vector<std::shared_ptr<arrow::Table>>& sources,
    std::vector<int64_t>& cursors,
    const std::vector<std::string>& struct_names,
    const std::shared_ptr<arrow::Schema>& schema)
{
    int64_t src_counts[3] = {};
    for (int64_t i = 0; i < chunk_len; ++i) ++src_counts[chunk_src[i]];

    std::vector<uint32_t> src_pos[3];
    for (int s = 0; s < 3; ++s) src_pos[s].reserve(src_counts[s]);
    for (int64_t i = 0; i < chunk_len; ++i)
        src_pos[chunk_src[i]].push_back(static_cast<uint32_t>(i));

    std::shared_ptr<arrow::Table> slices[3];
    for (int s = 0; s < 3; ++s) {
        slices[s] = sources[s]->Slice(cursors[s], src_counts[s]);
        cursors[s] += src_counts[s];
    }

    auto st_buf = unwrap(arrow::AllocateBuffer(chunk_len * sizeof(int64_t)));
    auto sc_buf = unwrap(arrow::AllocateBuffer(chunk_len * sizeof(int64_t)));
    auto tp_buf = unwrap(arrow::AllocateBuffer(chunk_len * sizeof(int8_t)));
    auto* st_out = reinterpret_cast<int64_t*>(st_buf->mutable_data());
    auto* sc_out = reinterpret_cast<int64_t*>(sc_buf->mutable_data());
    auto* tp_out = reinterpret_cast<int8_t*>(tp_buf->mutable_data());

    for (int s = 0; s < 3; ++s) {
        int64_t n = src_counts[s];
        if (n == 0) continue;
        auto st_src = raw_int64(combine_chunks(slices[s]->GetColumnByName("sort_time")));
        auto sc_src = raw_int64(combine_chunks(slices[s]->GetColumnByName("stock_code")));
        auto tp_src = std::static_pointer_cast<arrow::Int8Array>(
            combine_chunks(slices[s]->GetColumnByName("type_priority")))->raw_values();
        for (size_t k = 0; k < src_pos[s].size(); ++k) {
            auto p = src_pos[s][k];
            st_out[p] = st_src[k];
            sc_out[p] = sc_src[k];
            tp_out[p] = tp_src[k];
        }
    }

    auto sort_time_arr = std::make_shared<arrow::Int64Array>(chunk_len, std::move(st_buf));
    auto stock_code_arr = std::make_shared<arrow::Int64Array>(chunk_len, std::move(sc_buf));
    auto tp_arr = std::make_shared<arrow::Int8Array>(chunk_len, std::move(tp_buf));

    static const char* dt_labels[] = {"tick", "order", "trans"};
    arrow::StringBuilder dt_builder;
    ARROW_OK(dt_builder.Reserve(chunk_len));
    ARROW_OK(dt_builder.ReserveData(chunk_len * 5));
    for (int64_t i = 0; i < chunk_len; ++i)
        dt_builder.UnsafeAppend(std::string_view(dt_labels[tp_arr->Value(i)]));
    auto data_type_arr = unwrap(dt_builder.Finish(), "dt");

    auto lt_arr = unwrap(arrow::MakeArrayOfNull(arrow::int64(), chunk_len), "null_lt");

    std::shared_ptr<arrow::Array> struct_cols[3];
    for (int sid = 0; sid < 3; ++sid) {
        auto col_name = struct_names[sid];
        auto ftype = schema->GetFieldByName(col_name)->type();

        if (src_counts[sid] == 0) {
            struct_cols[sid] = unwrap(arrow::MakeArrayOfNull(ftype, chunk_len));
            continue;
        }

        auto indices = make_scatter_indices(src_pos[sid], chunk_len, src_counts[sid]);
        auto src_col = combine_chunks(slices[sid]->GetColumnByName(col_name));
        struct_cols[sid] = unwrap(
            arrow::compute::CallFunction("take", {src_col, indices}),
            "scatter_struct").make_array();
    }

    auto mc = [](std::shared_ptr<arrow::Array> a) {
        return std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{std::move(a)});
    };
    return arrow::Table::Make(schema, {
        mc(sort_time_arr), mc(tp_arr), mc(stock_code_arr),
        mc(data_type_arr), mc(lt_arr),
        mc(struct_cols[0]), mc(struct_cols[1]), mc(struct_cols[2]),
    });
}

// ── compute_local_time ───────────────────────────────────────────────

static std::vector<int64_t> compute_local_time(
    const int64_t* sort_time, int64_t n,
    std::optional<int64_t> prev_last)
{
    std::vector<int64_t> adj(n);
    for (int64_t i = 0; i < n; ++i) adj[i] = sort_time[i] - i;
    if (prev_last.has_value() && n > 0)
        adj[0] = std::max(adj[0], *prev_last + 1);
    for (int64_t i = 1; i < n; ++i)
        adj[i] = std::max(adj[i], adj[i - 1]);
    for (int64_t i = 0; i < n; ++i) adj[i] += i;
    return adj;
}

// ── SeqData (owns memory) ────────────────────────────────────────────

struct SeqData {
    std::vector<int64_t>  times;
    std::vector<int8_t>   types;
    std::vector<int64_t>  codes;
    std::vector<uint32_t> locs;

    int64_t size() const { return static_cast<int64_t>(times.size()); }

    MergeSequence as_seq() const {
        return {times.data(), types.data(), codes.data(), locs.data(), size()};
    }
};

// ══════════════════════════════════════════════════════════════════════
// Stage 2: NPY merge helpers
// ══════════════════════════════════════════════════════════════════════

static int64_t compute_cutoff_ns(const std::string& date_str) {
    int y = std::stoi(date_str.substr(0, 4));
    int m = std::stoi(date_str.substr(4, 2));
    int d = std::stoi(date_str.substr(6, 2));
    struct tm tm = {};
    tm.tm_year = y - 1900; tm.tm_mon = m - 1; tm.tm_mday = d;
    tm.tm_hour = 9; tm.tm_min = 40; tm.tm_sec = 0;
    // CST = UTC+8
    time_t utc = timegm(&tm) - 8 * 3600;
    return static_cast<int64_t>(utc) * 1'000'000'000LL;
}

// ── Hash key for joins ───────────────────────────────────────────────

struct JoinKey {
    int64_t a, b;
    bool operator==(const JoinKey& o) const { return a == o.a && b == o.b; }
};
struct JoinKeyHash {
    size_t operator()(const JoinKey& k) const {
        size_t h = std::hash<int64_t>{}(k.a);
        h ^= std::hash<int64_t>{}(k.b) * 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

// ── Per-type front join result ───────────────────────────────────────
// After joining with NPY, each type's front half is described by arrays
// that will feed into the K-way re-merge.

struct FrontJoined {
    int64_t n = 0;
    std::vector<int64_t>  local_time;
    std::vector<int8_t>   type_priority;
    std::vector<int64_t>  stock_code;
    std::vector<int32_t>  channel;   // -1 for tick
    std::vector<int64_t>  ch_seq;    //  0 for tick; SH: biz_index, SZ: order_index/trade_index (ApplSeq)
    std::vector<int64_t>  src_row;   // row index in the front_type Arrow table
};

// ── Tick right join ──────────────────────────────────────────────────
// NPY drives: output = all NPY rows that have matching parquet rows.
// Positional matching within each (ticker, exch_time) group.

static FrontJoined join_tick_front(
    const std::shared_ptr<arrow::Table>& front_tick,
    const npy::TickData& npy)
{
    auto t0 = now_sec();
    int64_t n_pq = front_tick->num_rows();

    // Extract join keys from tick_data struct
    auto struct_col = std::static_pointer_cast<arrow::StructArray>(
        combine_chunks(front_tick->GetColumnByName("tick_data")));
    auto pq_ticker = struct_field_i64(struct_col, "ticker");
    auto pq_et_field = struct_col->GetFieldByName("exch_time");
    // Cast exch_time to int64 for uniform key
    auto pq_et = struct_field_i64(struct_col, "exch_time");

    auto pq_sc = std::static_pointer_cast<arrow::Int64Array>(
        combine_chunks(front_tick->GetColumnByName("stock_code")));

    // Filter NPY to tickers present in parquet
    std::set<int64_t> pq_tickers_set;
    for (int64_t i = 0; i < n_pq; ++i) pq_tickers_set.insert(pq_ticker->Value(i));

    // Build parquet groups: (ticker, exch_time) → ordered list of row indices
    std::unordered_map<JoinKey, std::vector<int64_t>, JoinKeyHash> pq_groups;
    pq_groups.reserve(n_pq);
    for (int64_t i = 0; i < n_pq; ++i) {
        JoinKey key{pq_ticker->Value(i), pq_et->Value(i)};
        pq_groups[key].push_back(i);
    }

    int64_t n_npy_filtered = 0;
    int64_t n_npy_zero_field = 0;
    int64_t n_null_ticker_missing = 0;
    for (int64_t i = 0; i < npy.n; ++i) {
        if (npy.ticker[i] == 0 || npy.exch_time[i] == 0) { ++n_npy_zero_field; continue; }
        if (pq_tickers_set.count(npy.ticker[i]) == 0) { ++n_null_ticker_missing; continue; }
        ++n_npy_filtered;
    }
    LOG("  tick join: %ld pq rows, %ld/%ld npy rows after filter",
        n_pq, n_npy_filtered, npy.n);

    // Iterate NPY in file order (preserves NPY's natural local_time ordering,
    // matching the Python right-join which keeps the right-table order).
    // Per-group positional counter for duplicate (ticker, exch_time) keys.
    std::unordered_map<JoinKey, int64_t, JoinKeyHash> group_cursor;
    FrontJoined result;
    result.local_time.reserve(n_npy_filtered);
    result.type_priority.reserve(n_npy_filtered);
    result.stock_code.reserve(n_npy_filtered);
    result.channel.reserve(n_npy_filtered);
    result.ch_seq.reserve(n_npy_filtered);
    result.src_row.reserve(n_npy_filtered);

    int64_t n_dropped_pq = 0;
    int64_t n_npy_no_match = 0;
    for (int64_t i = 0; i < npy.n; ++i) {
        if (npy.ticker[i] == 0 || npy.exch_time[i] == 0) continue;
        if (pq_tickers_set.count(npy.ticker[i]) == 0) continue;
        JoinKey key{npy.ticker[i], static_cast<int64_t>(npy.exch_time[i])};
        auto pq_it = pq_groups.find(key);
        if (pq_it == pq_groups.end()) {
            ++n_npy_no_match;
            continue;
        }
        auto& pq_idx_vec = pq_it->second;
        int64_t pos = group_cursor[key]++;
        if (pos >= static_cast<int64_t>(pq_idx_vec.size())) continue;
        int64_t pi = pq_idx_vec[pos];
        result.local_time.push_back(npy.local_time[i]);
        result.type_priority.push_back(0);
        result.stock_code.push_back(pq_sc->Value(pi));
        result.channel.push_back(-1);
        result.ch_seq.push_back(0);
        result.src_row.push_back(pi);
    }
    result.n = static_cast<int64_t>(result.local_time.size());

    // Count dropped parquet rows (those with more duplicates than NPY)
    for (auto& [key, pq_idx_vec] : pq_groups) {
        auto it = group_cursor.find(key);
        int64_t used = (it != group_cursor.end()) ? it->second : 0;
        if (used < static_cast<int64_t>(pq_idx_vec.size()))
            n_dropped_pq += static_cast<int64_t>(pq_idx_vec.size()) - used;
    }

    int64_t total_null = n_npy_zero_field + n_null_ticker_missing + n_npy_no_match;
    LOG("  tick right-join null (NPY key not in parquet): %ld / %ld",
        total_null, npy.n);
    if (n_npy_zero_field > 0)
        LOG("    - ticker=0 or exch_time=0: %ld", n_npy_zero_field);
    if (n_null_ticker_missing > 0)
        LOG("    - ticker not in parquet: %ld", n_null_ticker_missing);
    if (n_npy_no_match > 0)
        LOG("    - (ticker,exch_time) key not in parquet: %ld", n_npy_no_match);
    if (n_dropped_pq > 0)
        LOG("  tick: %ld parquet rows dropped (more duplicates than NPY)", n_dropped_pq);
    LOG("  tick join: %ld output rows (%.1fs)", result.n, now_sec() - t0);
    return result;
}

// ── Tick left join (parquet drives) ──────────────────────────────────
// All parquet rows kept, NPY local_time used when matched.

static FrontJoined join_tick_front_left(
    const std::shared_ptr<arrow::Table>& front_tick,
    const npy::TickData& npy)
{
    auto t0 = now_sec();
    int64_t n_pq = front_tick->num_rows();

    auto struct_col = std::static_pointer_cast<arrow::StructArray>(
        combine_chunks(front_tick->GetColumnByName("tick_data")));
    auto pq_ticker = struct_field_i64(struct_col, "ticker");
    auto pq_et = struct_field_i64(struct_col, "exch_time");
    auto pq_sc = std::static_pointer_cast<arrow::Int64Array>(
        combine_chunks(front_tick->GetColumnByName("stock_code")));
    auto orig_lt_arr = std::static_pointer_cast<arrow::Int64Array>(
        combine_chunks(front_tick->GetColumnByName("local_time")));

    // Build NPY groups: (ticker, exch_time) → ordered list of local_times
    std::unordered_map<JoinKey, std::vector<int64_t>, JoinKeyHash> npy_groups;
    int64_t n_npy_zero_field = 0;
    for (int64_t i = 0; i < npy.n; ++i) {
        if (npy.ticker[i] == 0 || npy.exch_time[i] == 0) { ++n_npy_zero_field; continue; }
        JoinKey key{npy.ticker[i], static_cast<int64_t>(npy.exch_time[i])};
        npy_groups[key].push_back(npy.local_time[i]);
    }
    if (n_npy_zero_field > 0)
        LOG("  tick NPY: %ld rows with ticker=0 or exch_time=0", n_npy_zero_field);

    // Left join: iterate parquet (all rows kept)
    std::unordered_map<JoinKey, int64_t, JoinKeyHash> group_cursor;
    FrontJoined result;
    result.n = n_pq;
    result.local_time.resize(n_pq);
    result.type_priority.assign(n_pq, 0);
    result.stock_code.resize(n_pq);
    result.channel.assign(n_pq, -1);
    result.ch_seq.assign(n_pq, 0);
    result.src_row.resize(n_pq);

    int64_t n_matched = 0;
    int64_t n_null = 0;
    for (int64_t i = 0; i < n_pq; ++i) {
        JoinKey key{pq_ticker->Value(i), pq_et->Value(i)};
        auto npy_it = npy_groups.find(key);
        if (npy_it != npy_groups.end()) {
            int64_t pos = group_cursor[key]++;
            if (pos < static_cast<int64_t>(npy_it->second.size())) {
                result.local_time[i] = npy_it->second[pos];
                ++n_matched;
            } else {
                result.local_time[i] = orig_lt_arr->Value(i);
                ++n_null;
            }
        } else {
            result.local_time[i] = orig_lt_arr->Value(i);
            ++n_null;
        }
        result.stock_code[i] = pq_sc->Value(i);
        result.src_row[i] = i;
    }

    LOG("  tick left-join null (parquet key not in NPY): %ld / %ld",
        n_null, n_pq);
    LOG("  tick join: %ld rows, %ld/%ld npy matched (%.1fs)",
        n_pq, n_matched, npy.n, now_sec() - t0);
    return result;
}

// ── Trans / Order left join ──────────────────────────────────────────
// Parquet drives: all rows kept, NPY local_time used when available.

static FrontJoined join_channel_front(
    const std::shared_ptr<arrow::Table>& front_table,
    const npy::ChannelData& npy,
    int8_t type_pri,
    const std::string& struct_col_name,
    const std::string& index_field_name)
{
    auto t0 = now_sec();
    int64_t n_pq = front_table->num_rows();

    auto struct_col = std::static_pointer_cast<arrow::StructArray>(
        combine_chunks(front_table->GetColumnByName(struct_col_name)));
    auto pq_ch = struct_field_i64(struct_col, "channel");
    auto pq_idx = struct_field_i64(struct_col, index_field_name);

    // Read biz_index for SH merge key
    auto pq_biz = struct_field_i64(struct_col, "biz_index");

    auto pq_sc = std::static_pointer_cast<arrow::Int64Array>(
        combine_chunks(front_table->GetColumnByName("stock_code")));

    auto orig_lt_arr = std::static_pointer_cast<arrow::Int64Array>(
        combine_chunks(front_table->GetColumnByName("local_time")));

    // Build NPY lookup: (channel, index) → npy_local_time
    // NPY key always uses order_index/trade_index for the join
    std::unordered_map<JoinKey, int64_t, JoinKeyHash> npy_map;
    npy_map.reserve(npy.n);
    for (int64_t i = 0; i < npy.n; ++i) {
        npy_map[{static_cast<int64_t>(npy.channel[i]), npy.index[i]}] = npy.local_time[i];
    }

    FrontJoined result;
    result.n = n_pq;
    result.local_time.resize(n_pq);
    result.type_priority.assign(n_pq, type_pri);
    result.stock_code.resize(n_pq);
    result.channel.resize(n_pq);
    result.ch_seq.resize(n_pq);
    result.src_row.resize(n_pq);

    int64_t n_matched = 0;
    int64_t n_npy_no_match = 0;
    for (int64_t i = 0; i < n_pq; ++i) {
        JoinKey key{pq_ch->Value(i), pq_idx->Value(i)};
        auto it = npy_map.find(key);
        if (it != npy_map.end()) {
            result.local_time[i] = it->second;
            ++n_matched;
        } else {
            result.local_time[i] = orig_lt_arr->Value(i);
            ++n_npy_no_match;
        }
        result.stock_code[i] = pq_sc->Value(i);
        result.channel[i] = static_cast<int32_t>(pq_ch->Value(i));
        // ch_seq = merge key: biz_index for SH, order_index/trade_index for SZ
        int64_t biz = pq_biz->Value(i);
        result.ch_seq[i] = (biz != 0) ? biz : pq_idx->Value(i);
        result.src_row[i] = i;
    }

    const char* label = (type_pri == 1) ? "order" : "trans";
    LOG("  %s left-join null (parquet key not in NPY): %ld / %ld",
        label, n_npy_no_match, n_pq);
    LOG("  %s join: %ld rows, %ld/%ld npy matched (%.1fs)",
        label, n_pq, n_matched, npy.n, now_sec() - t0);
    return result;
}

// ── Front K-way re-merge ─────────────────────────────────────────────
// K-way merge tick + per-channel order + per-channel trans by (local_time, stock_code).
// Unlike Stage 1 back-half, order and trans per channel are SEPARATE sequences
// (no biz_index/applseq two-way merge within a channel).

struct FrontMergeResult {
    std::vector<int64_t> tick_indices, tick_lt;
    std::vector<int64_t> order_indices, order_lt;
    std::vector<int64_t> trans_indices, trans_lt;
    int64_t last_lt = 0;
};

// Per-sequence storage for the front K-way merge (owns its arrays).
struct FrontSeqData {
    std::vector<int64_t>  times;
    std::vector<int8_t>   types;   // all set to 0 so HeapEntry compares (time, code)
    std::vector<int64_t>  codes;
    std::vector<uint32_t> locs;    // index into flat front arrays
};

static FrontMergeResult kway_merge_front(
    FrontJoined& tick_j, FrontJoined& order_j, FrontJoined& trans_j)
{
    auto t0 = now_sec();

    int64_t n_total = tick_j.n + order_j.n + trans_j.n;

    // ── 1. Build per-channel per-type sequences ─────────────────────

    // Flat arrays that map merge-output position → (type_id, src_row)
    // We pack: tick rows at offset 0, order at tick_j.n, trans at tick_j.n+order_j.n
    std::vector<int8_t>  flat_type_id(n_total);
    std::vector<int64_t> flat_src_row(n_total);
    std::vector<int64_t> flat_lt(n_total);

    for (int64_t i = 0; i < tick_j.n; ++i) {
        flat_type_id[i] = 0; flat_src_row[i] = tick_j.src_row[i]; flat_lt[i] = tick_j.local_time[i];
    }
    int64_t off_o = tick_j.n;
    for (int64_t i = 0; i < order_j.n; ++i) {
        flat_type_id[off_o + i] = 1; flat_src_row[off_o + i] = order_j.src_row[i]; flat_lt[off_o + i] = order_j.local_time[i];
    }
    int64_t off_t = tick_j.n + order_j.n;
    for (int64_t i = 0; i < trans_j.n; ++i) {
        flat_type_id[off_t + i] = 2; flat_src_row[off_t + i] = trans_j.src_row[i]; flat_lt[off_t + i] = trans_j.local_time[i];
    }

    std::vector<FrontSeqData> sequences;

    // 1a. tick: single sequence
    {
        FrontSeqData s;
        s.times.resize(tick_j.n); s.types.assign(tick_j.n, 0);
        s.codes.resize(tick_j.n); s.locs.resize(tick_j.n);
        for (int64_t i = 0; i < tick_j.n; ++i) {
            s.times[i] = tick_j.local_time[i];
            s.codes[i] = tick_j.stock_code[i];
            s.locs[i]  = static_cast<uint32_t>(i); // flat offset = i
        }
        sequences.push_back(std::move(s));
    }

    // 1b. Group order rows by channel
    auto build_channel_seqs = [&](FrontJoined& j, int64_t flat_offset) {
        std::map<int32_t, std::vector<int64_t>> by_ch;
        for (int64_t i = 0; i < j.n; ++i)
            by_ch[j.channel[i]].push_back(i);

        for (auto& [ch, indices] : by_ch) {
            FrontSeqData s;
            s.times.resize(indices.size()); s.types.assign(indices.size(), 0);
            s.codes.resize(indices.size()); s.locs.resize(indices.size());
            for (size_t k = 0; k < indices.size(); ++k) {
                int64_t i = indices[k];
                s.times[k] = j.local_time[i];
                s.codes[k] = j.stock_code[i];
                s.locs[k]  = static_cast<uint32_t>(flat_offset + i);
            }
            sequences.push_back(std::move(s));
        }
    };

    build_channel_seqs(order_j, off_o);
    int n_order_ch = static_cast<int>(sequences.size()) - 1;  // minus tick
    build_channel_seqs(trans_j, off_t);
    int n_seqs = static_cast<int>(sequences.size());
    int n_trans_ch = n_seqs - 1 - n_order_ch;
    LOG("  front: %d sequences (1 tick + %d order-ch + %d trans-ch), %ld total rows",
        n_seqs, n_order_ch, n_trans_ch, n_total);

    // ── 2. K-way merge ──────────────────────────────────────────────

    std::vector<MergeSequence> merge_seqs(n_seqs);
    for (int i = 0; i < n_seqs; ++i) {
        merge_seqs[i].times  = sequences[i].times.data();
        merge_seqs[i].types  = sequences[i].types.data();
        merge_seqs[i].codes  = sequences[i].codes.data();
        merge_seqs[i].locs   = sequences[i].locs.data();
        merge_seqs[i].length = static_cast<int64_t>(sequences[i].times.size());
    }

    std::vector<uint8_t>  out_src(n_total);
    std::vector<uint32_t> out_loc(n_total);
    kway_merge(merge_seqs, out_src.data(), out_loc.data(), n_total);

    LOG("  front K-way merge done (%.1fs): %ld rows, heap size %d",
        now_sec() - t0, n_total, n_seqs);

    // ── 3. Dedup local_time (strictly increasing) ───────────────────

    std::vector<int64_t> merged_lt(n_total);
    for (int64_t i = 0; i < n_total; ++i)
        merged_lt[i] = flat_lt[out_loc[i]];
    for (int64_t i = 1; i < n_total; ++i)
        if (merged_lt[i] <= merged_lt[i - 1]) merged_lt[i] = merged_lt[i - 1] + 1;

    // ── 4. Split by type → FrontMergeResult ─────────────────────────

    FrontMergeResult result;
    for (int64_t i = 0; i < n_total; ++i) {
        uint32_t flat_idx = out_loc[i];
        int8_t   tid = flat_type_id[flat_idx];
        int64_t  sr  = flat_src_row[flat_idx];
        int64_t  lt  = merged_lt[i];
        switch (tid) {
            case 0: result.tick_indices.push_back(sr);  result.tick_lt.push_back(lt);  break;
            case 1: result.order_indices.push_back(sr); result.order_lt.push_back(lt); break;
            case 2: result.trans_indices.push_back(sr); result.trans_lt.push_back(lt); break;
        }
    }
    result.last_lt = merged_lt.empty() ? 0 : merged_lt.back();

    LOG("  front merge done (%.1fs): tick=%zu order=%zu trans=%zu, last_lt=%ld",
        now_sec() - t0,
        result.tick_indices.size(), result.order_indices.size(),
        result.trans_indices.size(), result.last_lt);
    return result;
}

// ── Unnest struct + write per-type output ────────────────────────────
// Flatten the struct column into top-level columns, replace inner
// local_time with the outer merged one.

static void write_type_output(
    const std::string& label,
    const std::string& struct_col_name,
    const std::shared_ptr<arrow::Table>& front_table,
    const std::vector<int64_t>& front_indices,
    const std::vector<int64_t>& front_lt,
    const std::shared_ptr<arrow::Table>& back_table,
    int64_t lt_offset,
    const std::string& output_path,
    const std::shared_ptr<arrow::Schema>& target_schema)
{
    auto t0 = now_sec();

    // Reorder front rows by merge indices
    auto front_reordered = [&]{
        std::vector<uint32_t> idx32(front_indices.size());
        for (size_t i = 0; i < front_indices.size(); ++i)
            idx32[i] = static_cast<uint32_t>(front_indices[i]);
        auto idx_arr = wrap_u32(idx32.data(), static_cast<int64_t>(idx32.size()));
        return unwrap(
            arrow::compute::CallFunction("take", {front_table, idx_arr}),
            "take_front").table();
    }();

    int64_t n_front = front_reordered->num_rows();
    int64_t n_back = back_table->num_rows();
    int64_t n_total = n_front + n_back;
    LOG("  %s: front=%ld back=%ld total=%ld", label.c_str(), n_front, n_back, n_total);

    auto out_schema = target_schema;

    // Open chunked FileWriter (same pattern as Stage 1)
    fs::create_directories(fs::path(output_path).parent_path());
    auto outfile = unwrap(arrow::io::FileOutputStream::Open(output_path), "open_type_out");
    auto wr_props = parquet::WriterProperties::Builder()
                        .compression(arrow::Compression::ZSTD)
                        ->compression_level(3)
                        ->max_row_group_length(CHUNK_SIZE)
                        ->build();
    auto ar_props = parquet::ArrowWriterProperties::Builder()
                        .set_use_threads(true)
                        ->store_schema()
                        ->build();
    auto writer = unwrap(parquet::arrow::FileWriter::Open(
        *out_schema, arrow::default_memory_pool(), outfile, wr_props, ar_props),
        "open_type_writer");

    int serial_col_idx = out_schema->GetFieldIndex("serial");
    int lt_col_idx     = out_schema->GetFieldIndex("local_time");
    int et_col_idx     = out_schema->GetFieldIndex("exchange_time");

    // Helper: unnest struct column → flat table matching target_schema.
    // Reorders columns, casts types, inserts local_time as UInt64, fixes serial.
    auto unnest_chunk = [&](const std::shared_ptr<arrow::Table>& tbl,
                            const int64_t* lt_data, int64_t n, int64_t serial_offset)
        -> std::shared_ptr<arrow::Table>
    {
        auto sa = std::static_pointer_cast<arrow::StructArray>(
            combine_chunks(tbl->GetColumnByName(struct_col_name)));
        auto flat_arrays = unwrap(sa->Flatten(arrow::default_memory_pool()), "flatten");
        auto st = std::static_pointer_cast<arrow::StructType>(sa->type());

        std::unordered_map<std::string, std::shared_ptr<arrow::Array>> src_map;
        for (int i = 0; i < st->num_fields(); ++i)
            src_map[st->field(i)->name()] = flat_arrays[i];

        arrow::ChunkedArrayVector columns(out_schema->num_fields());
        for (int ci = 0; ci < out_schema->num_fields(); ++ci) {
            auto target_field = out_schema->field(ci);
            const auto& col_name = target_field->name();
            auto target_type = target_field->type();

            std::shared_ptr<arrow::Array> arr;
            if (col_name == "local_time") {
                arr = wrap_u64(lt_data, n);
            } else if (col_name == "serial") {
                arrow::Int32Builder b;
                ARROW_OK(b.Reserve(n));
                for (int64_t i = 0; i < n; ++i)
                    b.UnsafeAppend(static_cast<int32_t>(serial_offset + i));
                arr = unwrap(b.Finish(), "serial");
            } else {
                auto it = src_map.find(col_name);
                if (it == src_map.end()) {
                    arr = unwrap(arrow::MakeArrayOfNull(target_type, n),
                                 ("null_" + col_name).c_str());
                } else {
                    arr = it->second;
                }
            }

            if (!arr->type()->Equals(target_type)) {
                if (target_type->id() == arrow::Type::FIXED_SIZE_LIST &&
                    arr->type()->id() == arrow::Type::LIST) {
                    auto fsl_type = std::static_pointer_cast<arrow::FixedSizeListType>(target_type);
                    auto list_arr = std::static_pointer_cast<arrow::ListArray>(arr);
                    const int64_t list_size = fsl_type->list_size();
                    const int64_t length    = list_arr->length();
                    // Slice values/null_bitmap to the logical range of this (possibly
                    // sliced) ListArray so the resulting FSL with offset=0 is correct.
                    auto values = list_arr->values()->Slice(
                        list_arr->value_offset(0), length * list_size);
                    if (!values->type()->Equals(fsl_type->value_type())) {
                        arrow::compute::CastOptions vo;
                        vo.to_type = fsl_type->value_type();
                        values = unwrap(arrow::compute::CallFunction("cast", {values}, &vo),
                                        "cast_fsl_val").make_array();
                    }
                    std::shared_ptr<arrow::Buffer> null_bitmap;
                    int64_t null_count = list_arr->null_count();
                    if (null_count != 0 && list_arr->null_bitmap()) {
                        if (list_arr->offset() != 0) {
                            null_bitmap = unwrap(arrow::internal::CopyBitmap(
                                arrow::default_memory_pool(),
                                list_arr->null_bitmap()->data(),
                                list_arr->offset(), length), "copy_fsl_nb");
                        } else {
                            null_bitmap = list_arr->null_bitmap();
                        }
                    }
                    arr = std::make_shared<arrow::FixedSizeListArray>(
                        target_type, length, values, null_bitmap, null_count);
                } else {
                    arrow::compute::CastOptions opts;
                    opts.to_type = target_type;
                    arr = unwrap(arrow::compute::CallFunction("cast", {arr}, &opts),
                                 ("cast_" + col_name).c_str()).make_array();
                }
            }
            columns[ci] = std::make_shared<arrow::ChunkedArray>(
                arrow::ArrayVector{arr});
        }
        return arrow::Table::Make(out_schema, columns);
    };

    // Double-buffered chunked write with on-the-fly local_time verification
    std::shared_ptr<arrow::Table> pending;
    arrow::Status write_status;
    std::thread write_thread;
    int64_t prev_lt = INT64_MIN;
    int64_t violations = 0;

    auto submit_chunk = [&](std::shared_ptr<arrow::Table> chunk) {
        auto lt_col = std::static_pointer_cast<arrow::UInt64Array>(
            combine_chunks(chunk->GetColumnByName("local_time")));
        for (int64_t i = 0; i < chunk->num_rows(); ++i) {
            int64_t v = static_cast<int64_t>(lt_col->Value(i));
            if (v <= prev_lt) ++violations;
            prev_lt = v;
        }
        if (write_thread.joinable()) {
            write_thread.join();
            if (!write_status.ok()) {
                LOG("WRITE ERROR: %s", write_status.ToString().c_str());
                std::exit(1);
            }
        }
        pending = std::move(chunk);
        write_status = arrow::Status::OK();
        write_thread = std::thread([&writer, &pending, &write_status]() {
            write_status = writer->WriteTable(*pending, pending->num_rows());
        });
    };

    // Write front in CHUNK_SIZE slices
    int64_t serial_base = 0;
    for (int64_t start = 0; start < n_front; start += CHUNK_SIZE) {
        int64_t len = std::min(CHUNK_SIZE, n_front - start);
        auto slice = front_reordered->Slice(start, len);
        submit_chunk(unnest_chunk(slice, front_lt.data() + start, len, serial_base));
        serial_base += len;
    }
    front_reordered.reset();

    // Write back in CHUNK_SIZE slices (compute local_time per-chunk)
    for (int64_t start = 0; start < n_back; start += CHUNK_SIZE) {
        int64_t len = std::min(CHUNK_SIZE, n_back - start);
        auto slice = back_table->Slice(start, len);
        auto orig_lt = std::static_pointer_cast<arrow::Int64Array>(
            combine_chunks(slice->GetColumnByName("local_time")));
        std::vector<int64_t> chunk_lt(len);
        for (int64_t i = 0; i < len; ++i)
            chunk_lt[i] = orig_lt->Value(i) + lt_offset;
        submit_chunk(unnest_chunk(slice, chunk_lt.data(), len, serial_base));
        serial_base += len;
    }

    // Flush final pending write
    if (write_thread.joinable()) {
        write_thread.join();
        if (!write_status.ok()) {
            LOG("WRITE ERROR: %s", write_status.ToString().c_str());
            std::exit(1);
        }
    }

    ARROW_OK(writer->Close());
    ARROW_OK(outfile->Close());

    if (violations > 0)
        LOG("  WARNING: %s local_time NOT strictly increasing: %ld violations",
            label.c_str(), violations);
    else
        LOG("  %s local_time strictly increasing", label.c_str());

    LOG("  %s written: %ld rows -> %s (%.1fs)",
        label.c_str(), n_total, output_path.c_str(), now_sec() - t0);
}

// ── Post-process: fix serial + per-stock split ───────────────────────

static void post_process_type(
    const std::string& label,
    const std::string& output_path,
    bool is_tick)
{
    auto t0 = now_sec();
    LOG("  [post] %s: splitting into per-stock files ...", label.c_str());

    // Read all.parquet (serial already fixed during write_type_output)
    auto infile = unwrap(arrow::io::ReadableFile::Open(output_path), "pp_open");
    std::unique_ptr<parquet::arrow::FileReader> reader;
    ARROW_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
    std::shared_ptr<arrow::Table> tbl;
    ARROW_OK(reader->ReadTable(&tbl));
    infile.reset(); reader.reset();
    int64_t n = tbl->num_rows();
    LOG("  [post] %s: read %ld rows (%.1fs)", label.c_str(), n, now_sec() - t0);

    auto out_dir = fs::path(output_path).parent_path();
    std::string part_col = is_tick ? "wind_code" : "symbol";

    auto part_arr = combine_chunks(tbl->GetColumnByName(part_col));
    if (!part_arr) {
        LOG("  [post] %s: no '%s' column, skipping split", label.c_str(), part_col.c_str());
        return;
    }

    auto arr_to_str = [](const std::shared_ptr<arrow::Array>& arr, int64_t i) -> std::string {
        auto tid = arr->type_id();
        if (tid == arrow::Type::STRING)
            return std::static_pointer_cast<arrow::StringArray>(arr)->GetString(i);
        if (tid == arrow::Type::LARGE_STRING)
            return std::static_pointer_cast<arrow::LargeStringArray>(arr)->GetString(i);
        if (tid == arrow::Type::BINARY) {
            auto sv = std::static_pointer_cast<arrow::BinaryArray>(arr)->GetView(i);
            return std::string(reinterpret_cast<const char*>(sv.data()), sv.size());
        }
        if (tid == arrow::Type::LARGE_BINARY) {
            auto sv = std::static_pointer_cast<arrow::LargeBinaryArray>(arr)->GetView(i);
            return std::string(reinterpret_cast<const char*>(sv.data()), sv.size());
        }
        if (tid == arrow::Type::INT64)
            return std::to_string(std::static_pointer_cast<arrow::Int64Array>(arr)->Value(i));
        if (tid == arrow::Type::INT32)
            return std::to_string(std::static_pointer_cast<arrow::Int32Array>(arr)->Value(i));
        if (tid == arrow::Type::INT16)
            return std::to_string(std::static_pointer_cast<arrow::Int16Array>(arr)->Value(i));
        if (tid == arrow::Type::INT8)
            return std::to_string(std::static_pointer_cast<arrow::Int8Array>(arr)->Value(i));
        return "<unsupported>";
    };

    // Build partition groups: key → row indices
    struct GroupInfo {
        std::string key;
        std::string filename;
        std::vector<uint32_t> indices;
    };
    std::map<std::string, size_t> key_to_gidx;
    std::vector<GroupInfo> groups;

    if (is_tick) {
        for (int64_t i = 0; i < n; ++i) {
            auto key = arr_to_str(part_arr, i);
            auto it = key_to_gidx.find(key);
            if (it == key_to_gidx.end()) {
                key_to_gidx[key] = groups.size();
                auto dot = key.find('.');
                std::string fname = (dot != std::string::npos)
                    ? key.substr(dot + 1) + key.substr(0, dot) + ".parquet"
                    : key + ".parquet";
                groups.push_back({key, fname, {}});
                it = key_to_gidx.find(key);
            }
            groups[it->second].indices.push_back(static_cast<uint32_t>(i));
        }
    } else {
        auto mkt_arr = combine_chunks(tbl->GetColumnByName("market"));
        for (int64_t i = 0; i < n; ++i) {
            std::string sym = arr_to_str(part_arr, i);
            std::string mkt = arr_to_str(mkt_arr, i);
            int64_t mv = 0;
            try { mv = std::stoll(mkt); } catch (...) {}
            std::string exch = (mv == 49) ? "SH" : (mv == 48) ? "SZ" : ("M" + mkt);
            std::string key = exch + sym;
            auto it = key_to_gidx.find(key);
            if (it == key_to_gidx.end()) {
                key_to_gidx[key] = groups.size();
                groups.push_back({key, key + ".parquet", {}});
                it = key_to_gidx.find(key);
            }
            groups[it->second].indices.push_back(static_cast<uint32_t>(i));
        }
    }

    LOG("  [post] %s: %ld groups built (%.1fs)", label.c_str(),
        (long)groups.size(), now_sec() - t0);

    // Parallel write: each thread picks groups from a shared atomic counter
    auto wr_props = parquet::WriterProperties::Builder()
                        .compression(arrow::Compression::ZSTD)
                        ->compression_level(3)
                        ->build();
    auto ar_props = parquet::ArrowWriterProperties::Builder()
                        .set_use_threads(false)
                        ->build();
    int serial_col = tbl->schema()->GetFieldIndex("serial");
    auto tbl_schema = tbl->schema();

    std::atomic<int64_t> next_g{0};
    std::atomic<int> ok_count{0};
    int64_t n_groups = static_cast<int64_t>(groups.size());
    int n_threads = std::min(16, static_cast<int>(n_groups));

    auto worker = [&]() {
        while (true) {
            int64_t g = next_g.fetch_add(1);
            if (g >= n_groups) break;

            auto& gi = groups[g];
            if (gi.indices.empty()) continue;

            auto idx_arr = wrap_u32(gi.indices.data(),
                                    static_cast<int64_t>(gi.indices.size()));
            auto part_tbl = unwrap(
                arrow::compute::CallFunction("take", {tbl, idx_arr}),
                "take_part").table();

            if (serial_col >= 0) {
                int64_t pn = part_tbl->num_rows();
                arrow::Int32Builder b;
                ARROW_OK(b.Reserve(pn));
                for (int32_t j = 0; j < static_cast<int32_t>(pn); ++j)
                    b.UnsafeAppend(j);
                auto sa = unwrap(b.Finish(), "part_serial");
                auto sc = std::make_shared<arrow::ChunkedArray>(
                    arrow::ArrayVector{sa});
                part_tbl = unwrap(part_tbl->SetColumn(serial_col,
                    tbl_schema->field(serial_col), sc), "set_part_serial");
            }

            auto path = (out_dir / gi.filename).string();
            auto pf = unwrap(arrow::io::FileOutputStream::Open(path), "pp_part");
            ARROW_OK(parquet::arrow::WriteTable(
                *part_tbl, arrow::default_memory_pool(), pf, CHUNK_SIZE,
                wr_props, ar_props));
            ARROW_OK(pf->Close());
            ok_count.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i)
        threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    LOG("  [post] %s split: %d stock files (%.1fs)",
        label.c_str(), ok_count.load(), now_sec() - t0);
}

// ── Output path helper ───────────────────────────────────────────────

static std::string get_output_path(const std::string& date_str,
                                   const std::string& quote_num,
                                   const std::string& output_base)
{
    std::string ym  = date_str.substr(0,4) + "." + date_str.substr(4,2);
    std::string ymd = ym + "." + date_str.substr(6,2);
    return output_base + "/" + ym + "/" + ymd + "/default/" + quote_num + "/all.parquet";
}

// ── CLI argument parsing ─────────────────────────────────────────────

struct Args {
    std::string date_str;
    std::string data_dir;
    std::string output_dir  = "./output";
    std::string npy_dir;
    std::string output_base = "./output/stg2";
    std::string log_file;
    bool write_intermediate = false;
    bool post_process       = false;
    bool delta_npy          = false;
    bool tick_left_join     = false;
    bool sh_front_merge    = false;
    bool tick_only          = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-dir"     && i+1 < argc) { a.output_dir  = argv[++i]; continue; }
        if (arg == "--npy-dir"        && i+1 < argc) { a.npy_dir     = argv[++i]; continue; }
        if (arg == "--output-base"    && i+1 < argc) { a.output_base = argv[++i]; continue; }
        if (arg == "--log-file"       && i+1 < argc) { a.log_file    = argv[++i]; continue; }
        if (arg == "--write-intermediate") { a.write_intermediate = true; continue; }
        if (arg == "--post-process")       { a.post_process = true; continue; }
        if (arg == "--delta-npy")          { a.delta_npy = true; continue; }
        if (arg == "--tick-left-join")     { a.tick_left_join = true; continue; }
        if (arg == "--sh-front-merge")    { a.sh_front_merge = true; continue; }
        if (arg == "--tick-only")          { a.tick_only = true; continue; }
        positional.push_back(arg);
    }

    if (positional.size() < 2) {
        fprintf(stderr,
            "Usage: %s <YYYYMMDD> <data_dir> [options]\n"
            "Options:\n"
            "  --output-dir DIR          Intermediate sorted parquet dir (default: ./output)\n"
            "  --npy-dir DIR             NPY base dir (enables stage 2 merge)\n"
            "  --output-base DIR         Stage 2 output base (default: ./output/stg2)\n"
            "  --write-intermediate      Also write sorted_{date}.parquet\n"
            "  --post-process            Fix serial + split per stock\n"
            "  --delta-npy               Use delta-encoded NPY format\n"
            "  --tick-left-join          Tick uses left-join (parquet drives, default: right-join/NPY drives)\n"
            "  --sh-front-merge         Front: merge SH order+trans per channel by biz_index (default: off)\n"
            "  --tick-only               Stage 2: only write tick output, skip trans & order\n"
            "  --log-file FILE           Also write log to FILE\n",
            argv[0]);
        std::exit(1);
    }

    a.date_str = positional[0];
    a.data_dir = positional[1];
    if (positional.size() >= 3 && a.output_dir == "./output")
        a.output_dir = positional[2];

    if (a.date_str.size() != 8) {
        fprintf(stderr, "Invalid date '%s', expected YYYYMMDD\n", a.date_str.c_str());
        std::exit(1);
    }

    // If npy_dir not specified and no --write-intermediate, default to write intermediate
    if (a.npy_dir.empty() && !a.write_intermediate) {
        a.write_intermediate = true;
    }

    return a;
}

// ══════════════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    auto args = parse_args(argc, argv);

    if (!args.log_file.empty()) {
        g_log_fp = fopen(args.log_file.c_str(), "w");
        if (!g_log_fp) {
            fprintf(stderr, "Cannot open log file '%s'\n", args.log_file.c_str());
            return 1;
        }
    }

    auto ts = now_sec();

    // ── 1. Read (three types with shared thread pool) ─────────────
    LOG("Reading parquet data from %s ...", args.data_dir.c_str());
    std::shared_ptr<arrow::Table> tick, trans, order;
    read_all_types(args.data_dir, tick, trans, order);

    // ── 2. Pre-sort (three types in parallel) ─────────────────────
    LOG("Pre-sorting ...");
    auto t0 = now_sec();
    {
        std::thread s1([&]{ tick  = sort_table(tick,  {"exchange_time", "ticker"}); });
        // SH: biz_index is the per-channel shared seq → sort by it.
        // SZ: order_index/trade_index is the per-channel ApplSeq.
        // Since both SH and SZ rows coexist, we sort by channel first,
        // then within each channel the correct seq key is already monotonic
        // after sorting by (channel, biz_index) for SH and (channel, order_index) for SZ.
        // For simplicity, sort by the seq key used for merging:
        //   order: pick biz_index if non-zero (SH), else order_index (SZ).
        //   trans: pick biz_index if non-zero (SH), else trade_index (SZ).
        // Since pre-sort is per-channel and biz_index=0 for SZ, we can just sort
        // by both keys combined: (channel, biz_index, order_index). For SH rows
        // biz_index>0 dominates; for SZ rows biz_index=0 so order_index dominates.
        std::thread s2([&]{ order = sort_table(order, {"channel", "biz_index", "order_index"}); });
        std::thread s3([&]{ trans = sort_table(trans, {"channel", "biz_index", "trade_index"}); });
        s1.join(); s2.join(); s3.join();
    }
    LOG("  done (%.1fs)", now_sec() - t0);

    int64_t n_tick  = tick->num_rows();
    int64_t n_order = order->num_rows();
    int64_t n_trans = trans->num_rows();
    int64_t total   = n_tick + n_order + n_trans;

    // ── 3. Extract key arrays + build sub-sequences ─────────────────
    LOG("Building sub-sequences ...");
    t0 = now_sec();

    auto tick_time_arr  = get_column(tick,  "exchange_time", arrow::int64());
    auto tick_code_arr  = get_column(tick,  "ticker",        arrow::int64());
    auto order_ch_arr   = get_column(order, "channel",       arrow::int32());
    auto order_time_arr = get_column(order, "exchange_time", arrow::int64());
    auto order_code_arr = get_column(order, "symbol",        arrow::int64());
    auto order_biz_raw  = get_column(order, "biz_index",     arrow::int64());
    auto order_idx_arr  = get_column(order, "order_index",   arrow::int64());
    auto trans_ch_arr   = get_column(trans, "channel",       arrow::int32());
    auto trans_time_arr = get_column(trans, "exchange_time", arrow::int64());
    auto trans_code_arr = get_column(trans, "symbol",        arrow::int64());
    auto trans_biz_raw  = get_column(trans, "biz_index",     arrow::int64());
    auto trans_idx_arr  = get_column(trans, "trade_index",   arrow::int64());

    const int64_t* tick_times  = raw_int64(tick_time_arr);
    const int64_t* tick_codes  = raw_int64(tick_code_arr);
    const int32_t* order_ch    = raw_int32(order_ch_arr);
    const int64_t* order_times = raw_int64(order_time_arr);
    const int64_t* order_codes = raw_int64(order_code_arr);
    const int32_t* trans_ch    = raw_int32(trans_ch_arr);
    const int64_t* trans_times = raw_int64(trans_time_arr);
    const int64_t* trans_codes = raw_int64(trans_code_arr);

    // Build per-row merge key: SH uses biz_index, SZ uses order_index/trade_index (ApplSeq).
    // biz_index == 0 for SZ rows, non-zero for SH rows.
    const int64_t* o_biz_raw = raw_int64(order_biz_raw);
    const int64_t* o_idx_raw = raw_int64(order_idx_arr);
    const int64_t* t_biz_raw = raw_int64(trans_biz_raw);
    const int64_t* t_idx_raw = raw_int64(trans_idx_arr);

    std::vector<int64_t> order_merge_key(n_order), trans_merge_key(n_trans);
    for (int64_t i = 0; i < n_order; ++i)
        order_merge_key[i] = (o_biz_raw[i] != 0) ? o_biz_raw[i] : o_idx_raw[i];
    for (int64_t i = 0; i < n_trans; ++i)
        trans_merge_key[i] = (t_biz_raw[i] != 0) ? t_biz_raw[i] : t_idx_raw[i];

    const int64_t* order_biz = order_merge_key.data();
    const int64_t* trans_biz = trans_merge_key.data();

    std::map<int32_t, std::vector<int64_t>> order_by_ch, trans_by_ch;
    for (int64_t i = 0; i < n_order; ++i) order_by_ch[order_ch[i]].push_back(i);
    for (int64_t i = 0; i < n_trans; ++i) trans_by_ch[trans_ch[i]].push_back(i);

    std::vector<SeqData> sequences;

    // 3a. tick: single sequence
    {
        SeqData s;
        s.times.resize(n_tick); s.types.assign(n_tick, 0);
        s.codes.resize(n_tick); s.locs.resize(n_tick);
        for (int64_t i = 0; i < n_tick; ++i) {
            s.times[i] = tick_times[i] * 1000;
            s.codes[i] = tick_codes[i];
            s.locs[i]  = static_cast<uint32_t>(i);
        }
        sequences.push_back(std::move(s));
    }

    // 3b. All channels: merge order+trans by sequence key per channel.
    //     SH (< 1000): biz_index is the shared sequence.
    //     SZ (>= 1000): order_index / trade_index share the same ApplSeq space.
    std::set<int32_t> merge_ch_set;
    for (auto& [ch, _] : order_by_ch) merge_ch_set.insert(ch);
    for (auto& [ch, _] : trans_by_ch) merge_ch_set.insert(ch);

    int64_t n_merged = 0;
    for (int32_t ch : merge_ch_set) {
        auto oit = order_by_ch.find(ch);
        auto tit = trans_by_ch.find(ch);
        const auto& o_idx = (oit != order_by_ch.end()) ? oit->second : std::vector<int64_t>{};
        const auto& t_idx = (tit != trans_by_ch.end()) ? tit->second : std::vector<int64_t>{};

        if (o_idx.empty() && t_idx.empty()) continue;

        if (o_idx.empty()) {
            SeqData s;
            for (auto i : t_idx) {
                s.times.push_back(trans_times[i] * 1000); s.types.push_back(2);
                s.codes.push_back(trans_codes[i]); s.locs.push_back(static_cast<uint32_t>(i));
            }
            n_merged += static_cast<int64_t>(t_idx.size());
            sequences.push_back(std::move(s));
            continue;
        }
        if (t_idx.empty()) {
            SeqData s;
            for (auto i : o_idx) {
                s.times.push_back(order_times[i] * 1000); s.types.push_back(1);
                s.codes.push_back(order_codes[i]); s.locs.push_back(static_cast<uint32_t>(i));
            }
            n_merged += static_cast<int64_t>(o_idx.size());
            sequences.push_back(std::move(s));
            continue;
        }

        // order_biz = biz_index for SH, order_index for SZ (both are the per-channel seq key)
        std::vector<int64_t> o_biz_v(o_idx.size()), t_biz_v(t_idx.size());
        for (size_t k = 0; k < o_idx.size(); ++k) o_biz_v[k] = order_biz[o_idx[k]];
        for (size_t k = 0; k < t_idx.size(); ++k) t_biz_v[k] = trans_biz[t_idx[k]];

        auto merge_order = two_way_merge_by_biz(
            o_biz_v.data(), static_cast<int64_t>(o_biz_v.size()),
            t_biz_v.data(), static_cast<int64_t>(t_biz_v.size()));

        int64_t no = static_cast<int64_t>(o_idx.size());
        SeqData s;
        s.times.resize(merge_order.size()); s.types.resize(merge_order.size());
        s.codes.resize(merge_order.size()); s.locs.resize(merge_order.size());
        for (size_t k = 0; k < merge_order.size(); ++k) {
            uint32_t mo = merge_order[k];
            if (mo < static_cast<uint32_t>(no)) {
                auto oi = o_idx[mo];
                s.times[k] = order_times[oi] * 1000; s.types[k] = 1;
                s.codes[k] = order_codes[oi]; s.locs[k] = static_cast<uint32_t>(oi);
            } else {
                auto ti = t_idx[mo - no];
                s.times[k] = trans_times[ti] * 1000; s.types[k] = 2;
                s.codes[k] = trans_codes[ti]; s.locs[k] = static_cast<uint32_t>(ti);
            }
        }
        n_merged += static_cast<int64_t>(merge_order.size());
        sequences.push_back(std::move(s));
    }

    int n_seqs = static_cast<int>(sequences.size());
    LOG("  %d sequences (%zu merged channels, %ld merged rows)  (%.1fs)",
        n_seqs, merge_ch_set.size(), n_merged, now_sec() - t0);

    order_by_ch.clear();
    trans_by_ch.clear();

    // ── 4. Pack struct payloads ──────────────────────────────────────
    LOG("Packing structs -> Arrow tables ...");
    t0 = now_sec();

    tick  = fix_array_cols(tick);
    order = fix_array_cols(order);
    trans = fix_array_cols(trans);

    auto tick_arrow  = build_meta_with_struct(tick,  "exchange_time", "ticker", 0, "tick",  "tick_data");
    tick.reset();
    auto order_arrow = build_meta_with_struct(order, "exchange_time", "symbol", 1, "order", "order_data");
    order.reset();
    auto trans_arrow = build_meta_with_struct(trans, "exchange_time", "symbol", 2, "trans", "trans_data");
    trans.reset();

    std::vector<std::shared_ptr<arrow::Table>> sources = {tick_arrow, order_arrow, trans_arrow};
    std::vector<std::string> struct_names = {"tick_data", "order_data", "trans_data"};
    auto schema = build_unified_schema(
        tick_arrow->schema(), order_arrow->schema(), trans_arrow->schema());

    LOG("  Arrow tables ready (%.1fs)", now_sec() - t0);
    const char* src_labels[] = {"tick", "order", "trans"};
    for (int i = 0; i < 3; ++i)
        LOG("  %5s : %12ld rows x %d cols",
            src_labels[i], sources[i]->num_rows(), sources[i]->num_columns());

    // ── 5. K-way merge ───────────────────────────────────────────────
    LOG("K-way merge (%ld rows, heap size %d) ...", total, n_seqs);
    t0 = now_sec();

    std::vector<MergeSequence> merge_seqs;
    merge_seqs.reserve(n_seqs);
    for (auto& sq : sequences) merge_seqs.push_back(sq.as_seq());

    std::vector<uint8_t>  merge_src(total);
    std::vector<uint32_t> merge_loc(total);

    kway_merge(merge_seqs, merge_src.data(), merge_loc.data(), total);

    double merge_sec = now_sec() - t0;
    LOG("  merge done in %.1fs  (%.0f rows/s)", merge_sec, total / merge_sec);

    sequences.clear(); sequences.shrink_to_fit();
    merge_seqs.clear();

    // ── 6a. Pre-reorder source tables ────────────────────────────────
    LOG("Pre-reordering source tables by merge order ...");
    t0 = now_sec();

    std::vector<uint32_t> locs_per[3];
    std::vector<int64_t>  pos_per[3];
    for (int i = 0; i < 3; ++i) {
        locs_per[i].reserve(sources[i]->num_rows());
        pos_per[i].reserve(sources[i]->num_rows());
    }
    for (int64_t j = 0; j < total; ++j) {
        int s = merge_src[j];
        locs_per[s].push_back(merge_loc[j]);
        pos_per[s].push_back(j);
    }
    merge_loc.clear(); merge_loc.shrink_to_fit();

    std::vector<int64_t> sort_times_all(total);
    {
        auto reorder_one = [&](int i) {
            if (locs_per[i].empty()) return;
            auto idx_arr = wrap_u32(locs_per[i].data(),
                                    static_cast<int64_t>(locs_per[i].size()));
            sources[i] = unwrap(
                arrow::compute::CallFunction("take", {sources[i], idx_arr}),
                "pre_reorder").table();
            auto st_data = raw_int64(
                combine_chunks(sources[i]->GetColumnByName("sort_time")));
            for (size_t k = 0; k < pos_per[i].size(); ++k)
                sort_times_all[pos_per[i][k]] = st_data[k];
            locs_per[i] = {};
            pos_per[i] = {};
        };
        std::thread r0([&]{ reorder_one(0); });
        std::thread r1([&]{ reorder_one(1); });
        std::thread r2([&]{ reorder_one(2); });
        r0.join(); r1.join(); r2.join();
    }

    LOG("  pre-reorder done in %.1fs", now_sec() - t0);

    // ══════════════════════════════════════════════════════════════════
    // Stage 1 output: write intermediate sorted parquet (optional)
    // ══════════════════════════════════════════════════════════════════

    if (args.write_intermediate) {
        fs::create_directories(args.output_dir);
        std::string out_path = args.output_dir + "/sorted_" + args.date_str + ".parquet";

        LOG("Chunked write (%ld rows, chunk %ld) -> %s ...",
            total, CHUNK_SIZE, out_path.c_str());
        t0 = now_sec();

        auto outfile = unwrap(arrow::io::FileOutputStream::Open(out_path), "open_out");
        auto wr_props = parquet::WriterProperties::Builder()
                            .compression(arrow::Compression::ZSTD)
                            ->compression_level(3)
                            ->max_row_group_length(CHUNK_SIZE)
                            ->build();
        auto ar_props = parquet::ArrowWriterProperties::Builder()
                            .set_use_threads(true)
                            ->build();
        auto writer = unwrap(parquet::arrow::FileWriter::Open(
            *schema, arrow::default_memory_pool(), outfile, wr_props, ar_props),
            "open_writer");

        std::optional<int64_t> prev_last_local;
        int64_t written = 0;
        int n_chunks = 0;
        std::vector<int64_t> cursors_wr = {0, 0, 0};
        int lt_idx = schema->GetFieldIndex("local_time");

        // Need a copy of sources for the chunked writer (it consumes cursors)
        auto sources_copy = sources; // shared_ptr copies are cheap

        auto build_chunk_wr = [&](int64_t start) -> std::shared_ptr<arrow::Table> {
            int64_t end = std::min(start + CHUNK_SIZE, total);
            int64_t chunk_len = end - start;
            auto batch_table = build_sorted_batch(
                merge_src.data() + start, chunk_len,
                sources_copy, cursors_wr, struct_names, schema);
            auto local_times = compute_local_time(
                sort_times_all.data() + start, chunk_len, prev_last_local);
            prev_last_local = local_times.back();
            auto lt_buf = unwrap(arrow::AllocateBuffer(chunk_len * sizeof(int64_t)));
            std::memcpy(lt_buf->mutable_data(), local_times.data(),
                        chunk_len * sizeof(int64_t));
            auto lt_arr = std::make_shared<arrow::Int64Array>(chunk_len, std::move(lt_buf));
            auto lt_chunked = std::make_shared<arrow::ChunkedArray>(
                arrow::ArrayVector{lt_arr});
            return unwrap(batch_table->SetColumn(
                lt_idx, arrow::field("local_time", arrow::int64()), lt_chunked),
                "set_lt");
        };

        std::shared_ptr<arrow::Table> pending_table;
        arrow::Status write_status;
        std::thread write_thread;

        for (int64_t start = 0; start < total; start += CHUNK_SIZE) {
            auto table = build_chunk_wr(start);
            int64_t chunk_len = std::min(CHUNK_SIZE, total - start);

            if (write_thread.joinable()) {
                write_thread.join();
                if (!write_status.ok()) {
                    LOG("WRITE ERROR: %s", write_status.ToString().c_str());
                    return 1;
                }
            }

            pending_table = std::move(table);
            write_status = arrow::Status::OK();
            write_thread = std::thread([&writer, &pending_table, &write_status]() {
                write_status = writer->WriteTable(*pending_table, pending_table->num_rows());
            });

            written += chunk_len;
            ++n_chunks;
        }

        if (write_thread.joinable()) {
            write_thread.join();
            if (!write_status.ok()) {
                LOG("WRITE ERROR: %s", write_status.ToString().c_str());
                return 1;
            }
        }

        ARROW_OK(writer->Close());
        ARROW_OK(outfile->Close());

        LOG("  intermediate write done in %.1fs  (%ld rows, %d chunks)",
            now_sec() - t0, written, n_chunks);
    }

    // ══════════════════════════════════════════════════════════════════
    // Stage 2: NPY merge + per-type output
    // ══════════════════════════════════════════════════════════════════

    if (!args.npy_dir.empty()) {
        LOG("============================================================");
        LOG("Stage 2: NPY merge (npy_dir=%s)", args.npy_dir.c_str());

        // 7. Compute local_time for all rows
        t0 = now_sec();
        auto local_times_all = compute_local_time(sort_times_all.data(), total, std::nullopt);
        LOG("  local_time computed for %ld rows (%.1fs)", total, now_sec() - t0);

        // Add local_time to each source table
        // sources[i] are in merge order; we need to set their local_time columns
        {
            int64_t cursor[3] = {};
            for (int64_t j = 0; j < total; ++j) {
                int s = merge_src[j];
                // We'll set local_time per-source below
                (void)s;
            }
            // Build per-source local_time arrays
            std::vector<int64_t> lt_per[3];
            for (int i = 0; i < 3; ++i)
                lt_per[i].reserve(sources[i]->num_rows());
            for (int64_t j = 0; j < total; ++j)
                lt_per[merge_src[j]].push_back(local_times_all[j]);

            for (int i = 0; i < 3; ++i) {
                auto lt_arr = wrap_i64(lt_per[i].data(),
                    static_cast<int64_t>(lt_per[i].size()));
                auto lt_chunked = std::make_shared<arrow::ChunkedArray>(
                    arrow::ArrayVector{lt_arr});
                // Find or add local_time column
                int lt_idx = sources[i]->schema()->GetFieldIndex("local_time");
                if (lt_idx >= 0) {
                    sources[i] = unwrap(sources[i]->SetColumn(lt_idx,
                        arrow::field("local_time", arrow::int64()), lt_chunked), "set_lt_src");
                } else {
                    sources[i] = unwrap(sources[i]->AddColumn(
                        sources[i]->num_columns(),
                        arrow::field("local_time", arrow::int64()), lt_chunked), "add_lt_src");
                }
            }
        }

        // 8. Split front/back
        int64_t cutoff_ns = compute_cutoff_ns(args.date_str);
        LOG("  cutoff_ns=%ld (09:40 CST)", cutoff_ns);

        // sort_times_all is in global merge order (ascending)
        int64_t n_front = static_cast<int64_t>(
            std::upper_bound(sort_times_all.begin(), sort_times_all.end(), cutoff_ns)
            - sort_times_all.begin());

        // Count per-type in front
        int64_t n_front_per[3] = {};
        for (int64_t j = 0; j < n_front; ++j) ++n_front_per[merge_src[j]];

        LOG("  front: %ld rows (tick=%ld, order=%ld, trans=%ld)",
            n_front, n_front_per[0], n_front_per[1], n_front_per[2]);
        LOG("  back:  %ld rows", total - n_front);

        auto front_tick  = sources[0]->Slice(0, n_front_per[0]);
        auto front_order = sources[1]->Slice(0, n_front_per[1]);
        auto front_trans = sources[2]->Slice(0, n_front_per[2]);

        auto back_tick  = sources[0]->Slice(n_front_per[0]);
        auto back_order = sources[1]->Slice(n_front_per[1]);
        auto back_trans = sources[2]->Slice(n_front_per[2]);

        // 9. Load NPY
        t0 = now_sec();
        LOG("  Loading NPY files (%s mode) ...", args.delta_npy ? "delta" : "classic");
        npy::TickData    tick_npy;
        npy::ChannelData trans_npy, order_npy;
        if (args.delta_npy) {
            tick_npy  = npy::load_tick(args.npy_dir + "/snapshot.npy");
            trans_npy = npy::load_delta_channel_dir(args.npy_dir, "trans");
            order_npy = npy::load_delta_channel_dir(args.npy_dir, "order");
        } else {
            tick_npy  = npy::load_tick(
                args.npy_dir + "/" + args.date_str + "/snapshot.npy");
            trans_npy = npy::load_channel_dir(
                args.npy_dir + "/" + args.date_str + "/trans_localtime_by_channel");
            order_npy = npy::load_channel_dir(
                args.npy_dir + "/" + args.date_str + "/order_localtime_by_channel");
        }
        LOG("  NPY loaded: tick=%ld trans=%ld order=%ld (%.1fs)",
            tick_npy.n, trans_npy.n, order_npy.n, now_sec() - t0);

        // 10. Join front with NPY
        LOG("  Joining front with NPY (tick=%s-join) ...",
            args.tick_left_join ? "left" : "right");
        auto tick_joined = args.tick_left_join
            ? join_tick_front_left(front_tick, tick_npy)
            : join_tick_front(front_tick, tick_npy);
        auto order_joined = join_channel_front(front_order, order_npy, 1,
                                               "order_data", "order_index");
        auto trans_joined = join_channel_front(front_trans, trans_npy, 2,
                                               "trans_data", "trade_index");

        // 11. Sort front by (local_time, stock_code)
        LOG("  Sort front (sh_front_merge=%s) ...",
            args.sh_front_merge ? "on" : "off");
        auto front_result = kway_merge_front(tick_joined, order_joined, trans_joined);

        // 12. Compute offset
        int64_t first_back_lt = local_times_all[n_front];
        int64_t offset = front_result.last_lt + 1 - first_back_lt;
        LOG("  offset: last_front_lt=%ld, first_back_lt=%ld, offset=%+ld",
            front_result.last_lt, first_back_lt, offset);

        // 13. Per-type output
        LOG("============================================================");
        LOG("Stage 2: Per-type output");

        std::string paths[3];
        paths[0] = get_output_path(args.date_str, "1", args.output_base); // tick
        paths[1] = get_output_path(args.date_str, "2", args.output_base); // trans
        paths[2] = get_output_path(args.date_str, "3", args.output_base); // order

        write_type_output("tick", "tick_data",
            front_tick, front_result.tick_indices, front_result.tick_lt,
            back_tick, offset, paths[0], make_tick_schema());

        if (!args.tick_only) {
            write_type_output("trans", "trans_data",
                front_trans, front_result.trans_indices, front_result.trans_lt,
                back_trans, offset, paths[1], make_trans_schema());

            write_type_output("order", "order_data",
                front_order, front_result.order_indices, front_result.order_lt,
                back_order, offset, paths[2], make_order_schema());
        } else {
            LOG("  --tick-only: skipping trans & order output");
        }

        // 14. Post-process (optional)
        if (args.post_process) {
            LOG("============================================================");
            LOG("Post-process: fix serial + split stocks");
            post_process_type("tick",  paths[0], true);
            if (!args.tick_only) {
                post_process_type("trans", paths[1], false);
                post_process_type("order", paths[2], false);
            }
        }
    }

    // ── Summary ──────────────────────────────────────────────────────
    double total_sec = now_sec() - ts;
    LOG("============================================================");
    LOG("Total rows         : %ld", total);
    LOG("Elapsed            : %.1fs (%.1f min)", total_sec, total_sec / 60.0);

    if (g_log_fp) { fclose(g_log_fp); g_log_fp = nullptr; }
    return 0;
}
