// parquet_sort.cpp  –  K-way merge sort for tick / order / trans parquet quote data.
//
// Sorting rules:
//   1. exchange_time ascending (nanoseconds)
//   2. type priority   (tick=0 < order=1 < trans=2)
//   3. stock code asc
//   For SH channels: biz_index ordering takes priority within each channel.
//   For SZ channels: ApplSeq (order_index/trade_index) ordering within each channel.
//
// Output: per-type parquet with local_time = exchange_time.
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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kway_merge.h"

namespace fs = std::filesystem;

// ── Constants ────────────────────────────────────────────────────────

static constexpr int64_t CHUNK_SIZE           = 20'000'000;

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
        if (tid == arrow::Type::BINARY || tid == arrow::Type::LARGE_BINARY)
            new_type = arrow::large_utf8();
        else if (tid == arrow::Type::STRING || tid == arrow::Type::LARGE_STRING)
            new_type = arrow::large_utf8();
        else if (tid == arrow::Type::LIST) {
            auto lt = std::static_pointer_cast<arrow::ListType>(fld->type());
            new_type = arrow::large_list(lt->value_type());
        }
        else if (tid == arrow::Type::LARGE_LIST)
            continue;
        else if (tid == arrow::Type::FIXED_SIZE_LIST) {
            auto flt = std::static_pointer_cast<arrow::FixedSizeListType>(fld->type());
            new_type = arrow::large_list(flt->value_type());
        }
        else
            continue;

        if (fld->type()->Equals(new_type)) continue;

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

// ── Per-type output: unnest struct, set local_time=exchange_time ──────

static void write_type_output_simple(
    const std::string& label,
    const std::string& struct_col_name,
    const std::shared_ptr<arrow::Table>& source_table,
    const std::vector<int64_t>& local_time,   // aligned with source_table rows
    const std::string& output_path,
    const std::shared_ptr<arrow::Schema>& target_schema)
{
    auto t0 = now_sec();
    int64_t n_total = source_table->num_rows();
    LOG("  %s: %ld rows -> %s", label.c_str(), n_total, output_path.c_str());

    if (n_total == 0) {
        LOG("  %s: no rows, skipping", label.c_str());
        return;
    }

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
        *target_schema, arrow::default_memory_pool(), outfile, wr_props, ar_props),
        "open_type_writer");

    // Unnest struct + build output table matching target_schema
    auto unnest_chunk = [&](const std::shared_ptr<arrow::Table>& tbl,
                            int64_t n, int64_t serial_offset, const int64_t* lt_data)
        -> std::shared_ptr<arrow::Table>
    {
        auto sa = std::static_pointer_cast<arrow::StructArray>(
            combine_chunks(tbl->GetColumnByName(struct_col_name)));
        auto flat_arrays = unwrap(sa->Flatten(arrow::default_memory_pool()), "flatten");
        auto st = std::static_pointer_cast<arrow::StructType>(sa->type());

        std::unordered_map<std::string, std::shared_ptr<arrow::Array>> src_map;
        for (int i = 0; i < st->num_fields(); ++i)
            src_map[st->field(i)->name()] = flat_arrays[i];

        arrow::ChunkedArrayVector columns(target_schema->num_fields());
        for (int ci = 0; ci < target_schema->num_fields(); ++ci) {
            auto target_field = target_schema->field(ci);
            const auto& col_name = target_field->name();
            auto target_type = target_field->type();

            std::shared_ptr<arrow::Array> arr;
            if (col_name == "local_time") {
                // local_time derived from exchange_time with +1 dedup (precomputed)
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
        return arrow::Table::Make(target_schema, columns);
    };

    // Chunked write
    std::shared_ptr<arrow::Table> pending;
    arrow::Status write_status;
    std::thread write_thread;
    int64_t serial_base = 0;

    for (int64_t start = 0; start < n_total; start += CHUNK_SIZE) {
        int64_t len = std::min(CHUNK_SIZE, n_total - start);
        auto slice = source_table->Slice(start, len);
        auto chunk = unnest_chunk(slice, len, serial_base, local_time.data() + start);
        serial_base += len;

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
    }

    if (write_thread.joinable()) {
        write_thread.join();
        if (!write_status.ok()) {
            LOG("WRITE ERROR: %s", write_status.ToString().c_str());
            std::exit(1);
        }
    }
    ARROW_OK(writer->Close());
    ARROW_OK(outfile->Close());

    LOG("  %s written: %ld rows (%.1fs)", label.c_str(), n_total, now_sec() - t0);
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
    std::string output_base = "./output";
    std::string log_file;
    bool write_intermediate = false;
    bool post_process       = false;
    bool tick_only          = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-dir"     && i+1 < argc) { a.output_dir  = argv[++i]; continue; }
        if (arg == "--output-base"    && i+1 < argc) { a.output_base = argv[++i]; continue; }
        if (arg == "--log-file"       && i+1 < argc) { a.log_file    = argv[++i]; continue; }
        if (arg == "--write-intermediate") { a.write_intermediate = true; continue; }
        if (arg == "--post-process")       { a.post_process = true; continue; }
        if (arg == "--tick-only")          { a.tick_only = true; continue; }
        positional.push_back(arg);
    }

    if (positional.size() < 2) {
        fprintf(stderr,
            "Usage: %s <YYYYMMDD> <data_dir> [options]\n"
            "Options:\n"
            "  --output-dir DIR          Intermediate sorted parquet dir (default: ./output)\n"
            "  --output-base DIR         Per-type output base dir (default: ./output)\n"
            "  --write-intermediate      Also write sorted_{date}.parquet\n"
            "  --post-process            Fix serial + split per stock\n"
            "  --tick-only               Only write tick output, skip trans & order\n"
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
            // local_time = exchange_time (sort_times_all already in ns)
            auto lt_buf = unwrap(arrow::AllocateBuffer(chunk_len * sizeof(int64_t)));
            std::memcpy(lt_buf->mutable_data(), sort_times_all.data() + start,
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
    // Per-type output
    // ══════════════════════════════════════════════════════════════════

    {
        LOG("============================================================");
        LOG("Per-type output (local_time = exchange_time, +1 on ties)");
        t0 = now_sec();

        // Compute local_time in global merge order: base is exchange_time in
        // NANOSECONDS (sort_times_all is already exchange_time_us * 1000),
        // strictly increasing — if a value is <= previous, bump to previous + 1
        // (i.e. +1 nanosecond on ties), matching the legacy pipeline unit.
        std::vector<int64_t> lt_global(total);
        for (int64_t j = 0; j < total; ++j)
            lt_global[j] = sort_times_all[j]; // already ns (exchange_time_us * 1000)
        for (int64_t j = 1; j < total; ++j)
            if (lt_global[j] <= lt_global[j - 1])
                lt_global[j] = lt_global[j - 1] + 1;

        // Split per type (aligned with reordered sources[i] rows)
        std::vector<int64_t> lt_per[3];
        for (int i = 0; i < 3; ++i)
            lt_per[i].reserve(sources[i]->num_rows());
        for (int64_t j = 0; j < total; ++j)
            lt_per[merge_src[j]].push_back(lt_global[j]);
        lt_global.clear(); lt_global.shrink_to_fit();

        std::string paths[3];
        paths[0] = get_output_path(args.date_str, "1", args.output_base); // tick
        paths[1] = get_output_path(args.date_str, "2", args.output_base); // trans
        paths[2] = get_output_path(args.date_str, "3", args.output_base); // order

        write_type_output_simple("tick", "tick_data",
            sources[0], lt_per[0], paths[0], make_tick_schema());

        if (!args.tick_only) {
            write_type_output_simple("trans", "trans_data",
                sources[2], lt_per[2], paths[1], make_trans_schema());
            write_type_output_simple("order", "order_data",
                sources[1], lt_per[1], paths[2], make_order_schema());
        } else {
            LOG("  --tick-only: skipping trans & order output");
        }

        if (args.post_process) {
            LOG("============================================================");
            LOG("Post-process: fix serial + split stocks");
            post_process_type("tick",  paths[0], true);
            if (!args.tick_only) {
                post_process_type("trans", paths[1], false);
                post_process_type("order", paths[2], false);
            }
        }

        LOG("  per-type output done in %.1fs", now_sec() - t0);
    }

    // ── Summary ──────────────────────────────────────────────────────
    double total_sec = now_sec() - ts;
    LOG("============================================================");
    LOG("Total rows         : %ld", total);
    LOG("Elapsed            : %.1fs (%.1f min)", total_sec, total_sec / 60.0);

    if (g_log_fp) { fclose(g_log_fp); g_log_fp = nullptr; }
    return 0;
}

