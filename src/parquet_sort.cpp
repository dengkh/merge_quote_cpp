// parquet_sort.cpp  –  C++ port of parquet_sort_modified.py
//
// K-way merge sort for tick / order / trans parquet quote data.
//
// Sorting rules (identical to Python version):
//   1. time ascending  (exchange_time in nanoseconds)
//   2. type priority   (tick=0 < order=1 < trans=2)
//   3. stock code asc
//   For SH channels: biz_index ordering takes priority within each channel.
//
// Output schema:
//   sort_time, type_priority, stock_code, data_type, local_time,
//   tick_data(struct|null), order_data(struct|null), trans_data(struct|null)
//
// Build:
//   mkdir build && cd build && cmake .. && make -j

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "kway_merge.h"

namespace fs = std::filesystem;

// ── Constants ────────────────────────────────────────────────────────

static constexpr int64_t CHUNK_SIZE          = 20'000'000;
static constexpr int32_t SH_CHANNEL_THRESHOLD = 1000;

// ── Logging ──────────────────────────────────────────────────────────

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

// Get single contiguous Array from a ChunkedArray column.
static std::shared_ptr<arrow::Array> combine_chunks(
    const std::shared_ptr<arrow::ChunkedArray>& col)
{
    if (col->num_chunks() == 1) return col->chunk(0);
    return unwrap(arrow::Concatenate(col->chunks(), arrow::default_memory_pool()),
                  "combine_chunks");
}

// Get column by name, optionally cast.
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
        // binary/large_binary → numeric requires intermediate utf8 step
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

// Raw int64 pointer from an Int64Array.
static const int64_t* raw_int64(const std::shared_ptr<arrow::Array>& a) {
    return std::static_pointer_cast<arrow::Int64Array>(a)->raw_values();
}
static const int32_t* raw_int32(const std::shared_ptr<arrow::Array>& a) {
    return std::static_pointer_cast<arrow::Int32Array>(a)->raw_values();
}

// Build a constant-value array.
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

// Build Arrow UInt32Array from a raw pointer.
static std::shared_ptr<arrow::Array> wrap_u32(const uint32_t* data, int64_t n) {
    auto buf = unwrap(arrow::AllocateBuffer(n * sizeof(uint32_t)));
    std::memcpy(buf->mutable_data(), data, n * sizeof(uint32_t));
    return std::make_shared<arrow::UInt32Array>(n, std::move(buf));
}

// Promote binary→large_binary, utf8→large_utf8, list→large_list so
// all per-stock files share one schema before concatenation.
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

// ── Step 1 : Read parquet files ──────────────────────────────────────

static constexpr int READ_THREADS = 16;

static std::shared_ptr<arrow::Table> read_data(
    const std::string& data_dir, const std::string& sub_dir, const char* label)
{
    auto t0 = now_sec();
    std::string folder = data_dir + "/" + sub_dir;

    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(folder)) {
        if (e.path().extension() == ".parquet" &&
            e.path().filename() != "all.parquet")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    int n_files = static_cast<int>(files.size());
    int n_threads = std::min(READ_THREADS, n_files);

    // Each thread reads its slice of files into its own vector
    std::vector<std::vector<std::shared_ptr<arrow::Table>>> per_thread(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        int lo = static_cast<int>(int64_t(t) * n_files / n_threads);
        int hi = static_cast<int>(int64_t(t + 1) * n_files / n_threads);
        per_thread[t].reserve(hi - lo);

        threads.emplace_back([&files, &per_thread, t, lo, hi]() {
            auto pool = arrow::default_memory_pool();
            for (int i = lo; i < hi; ++i) {
                auto infile = unwrap(arrow::io::ReadableFile::Open(files[i]), "open");
                std::unique_ptr<parquet::arrow::FileReader> reader;
                ARROW_OK(parquet::arrow::OpenFile(infile, pool, &reader));
                std::shared_ptr<arrow::Table> tbl;
                ARROW_OK(reader->ReadTable(&tbl));
                per_thread[t].push_back(promote_to_large(std::move(tbl)));
            }
        });
    }
    for (auto& th : threads) th.join();

    // Flatten results preserving file order
    auto pool = arrow::default_memory_pool();
    std::vector<std::shared_ptr<arrow::Table>> tables;
    tables.reserve(n_files);
    for (auto& v : per_thread)
        tables.insert(tables.end(),
                      std::make_move_iterator(v.begin()),
                      std::make_move_iterator(v.end()));

    auto combined = unwrap(arrow::ConcatenateTables(tables, {}, pool), "concat");
    combined = unwrap(combined->CombineChunks(pool), "combine");

    LOG("[%s] %ld rows from %zu files (%.1fs, %d threads)",
        label, combined->num_rows(), files.size(), now_sec() - t0, n_threads);
    return combined;
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

    // sort_time = time_col * 1000
    auto time_arr = get_column(table, time_col, arrow::int64());
    auto scalar_1000 = arrow::MakeScalar(int64_t(1000));
    auto sort_time = unwrap(
        arrow::compute::CallFunction("multiply", {time_arr, scalar_1000}),
        "multiply").make_array();

    // stock_code = code_col as int64
    auto stock_code = get_column(table, code_col, arrow::int64());

    // constant columns
    auto type_priority = make_const_int8(n, type_pri);
    auto data_type = make_const_string(n, dtype_str);

    // struct from all original columns
    arrow::ArrayVector struct_arrays;
    arrow::FieldVector struct_fields;
    for (int i = 0; i < table->num_columns(); ++i) {
        struct_fields.push_back(table->schema()->field(i));
        struct_arrays.push_back(combine_chunks(table->column(i)));
    }
    auto struct_type = arrow::struct_(struct_fields);
    auto struct_arr = unwrap(
        arrow::StructArray::Make(struct_arrays, struct_fields), "make_struct");

    // assemble output table
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

// Build a UInt32 index array with null at positions NOT in `positions`.
// Valid entries: positions[k] → value k  (sequential source index).
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
    // Count per source and collect positions
    int64_t src_counts[3] = {};
    for (int64_t i = 0; i < chunk_len; ++i) ++src_counts[chunk_src[i]];

    std::vector<uint32_t> src_pos[3];
    for (int s = 0; s < 3; ++s) src_pos[s].reserve(src_counts[s]);
    for (int64_t i = 0; i < chunk_len; ++i)
        src_pos[chunk_src[i]].push_back(static_cast<uint32_t>(i));

    // Slice sources
    std::shared_ptr<arrow::Table> slices[3];
    for (int s = 0; s < 3; ++s) {
        slices[s] = sources[s]->Slice(cursors[s], src_counts[s]);
        cursors[s] += src_counts[s];
    }

    // ── Metadata: direct scatter into raw buffers ──────────────────
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

    // data_type: derive from type_priority
    static const char* dt_labels[] = {"tick", "order", "trans"};
    arrow::StringBuilder dt_builder;
    ARROW_OK(dt_builder.Reserve(chunk_len));
    ARROW_OK(dt_builder.ReserveData(chunk_len * 5));
    for (int64_t i = 0; i < chunk_len; ++i)
        dt_builder.UnsafeAppend(std::string_view(dt_labels[tp_arr->Value(i)]));
    auto data_type_arr = unwrap(dt_builder.Finish(), "dt");

    // local_time: null placeholder
    auto lt_arr = unwrap(arrow::MakeArrayOfNull(arrow::int64(), chunk_len), "null_lt");

    // ── Struct columns: Take from individual source with null-padded indices ──
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

    // ── Assemble output table ──────────────────────────────────────
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

// ── Sequence data (owns memory) ──────────────────────────────────────

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

// ── main ─────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <YYYYMMDD> <data_dir> [output_dir]\n"
            "  data_dir:   directory containing sub-dirs 1/ 2/ 3/\n"
            "  output_dir: where sorted_YYYYMMDD.parquet is written (default: ./output)\n",
            argv[0]);
        return 1;
    }

    std::string date_str = argv[1];
    std::string data_dir = argv[2];
    std::string out_dir  = (argc > 3) ? argv[3] : "./output";

    if (date_str.size() != 8) {
        fprintf(stderr, "Invalid date '%s', expected YYYYMMDD\n", date_str.c_str());
        return 1;
    }

    auto ts = now_sec();

    // ── 1. Read (three types in parallel) ──────────────────────────
    LOG("Reading parquet data from %s ...", data_dir.c_str());
    std::shared_ptr<arrow::Table> tick, trans, order;
    {
        std::thread t1([&]{ tick  = read_data(data_dir, "1", "tick "); });
        std::thread t2([&]{ trans = read_data(data_dir, "2", "trans"); });
        std::thread t3([&]{ order = read_data(data_dir, "3", "order"); });
        t1.join(); t2.join(); t3.join();
    }
    LOG("All reads done (%.1fs wall)", now_sec() - ts);

    // ── 2. Pre-sort (three types in parallel) ─────────────────────
    LOG("Pre-sorting ...");
    auto t0 = now_sec();
    {
        std::thread s1([&]{ tick  = sort_table(tick,  {"exchange_time", "ticker"}); });
        std::thread s2([&]{ order = sort_table(order, {"channel", "order_index"}); });
        std::thread s3([&]{ trans = sort_table(trans, {"channel", "trade_index"}); });
        s1.join(); s2.join(); s3.join();
    }
    LOG("  done (%.1fs)", now_sec() - t0);

    int64_t n_tick  = tick->num_rows();
    int64_t n_order = order->num_rows();
    int64_t n_trans = trans->num_rows();
    int64_t total   = n_tick + n_order + n_trans;

    // ── 3. Extract key arrays ────────────────────────────────────────
    LOG("Building sub-sequences ...");
    t0 = now_sec();

    auto tick_time_arr  = get_column(tick,  "exchange_time", arrow::int64());
    auto tick_code_arr  = get_column(tick,  "ticker",        arrow::int64());
    auto order_ch_arr   = get_column(order, "channel",       arrow::int32());
    auto order_time_arr = get_column(order, "exchange_time", arrow::int64());
    auto order_code_arr = get_column(order, "symbol",        arrow::int64());
    auto order_biz_arr  = get_column(order, "order_index",   arrow::int64());
    auto trans_ch_arr   = get_column(trans, "channel",       arrow::int32());
    auto trans_time_arr = get_column(trans, "exchange_time", arrow::int64());
    auto trans_code_arr = get_column(trans, "symbol",        arrow::int64());
    auto trans_biz_arr  = get_column(trans, "trade_index",   arrow::int64());

    const int64_t* tick_times  = raw_int64(tick_time_arr);
    const int64_t* tick_codes  = raw_int64(tick_code_arr);
    const int32_t* order_ch    = raw_int32(order_ch_arr);
    const int64_t* order_times = raw_int64(order_time_arr);
    const int64_t* order_codes = raw_int64(order_code_arr);
    const int64_t* order_biz   = raw_int64(order_biz_arr);
    const int32_t* trans_ch    = raw_int32(trans_ch_arr);
    const int64_t* trans_times = raw_int64(trans_time_arr);
    const int64_t* trans_codes = raw_int64(trans_code_arr);
    const int64_t* trans_biz   = raw_int64(trans_biz_arr);

    // Build channel → row-index maps (O(n) vs per-channel scans)
    std::map<int32_t, std::vector<int64_t>> order_by_ch, trans_by_ch;
    for (int64_t i = 0; i < n_order; ++i) order_by_ch[order_ch[i]].push_back(i);
    for (int64_t i = 0; i < n_trans; ++i) trans_by_ch[trans_ch[i]].push_back(i);

    std::vector<SeqData> sequences;

    // 3a. tick: single sequence
    {
        SeqData s;
        s.times.resize(n_tick);
        s.types.assign(n_tick, 0);
        s.codes.resize(n_tick);
        s.locs.resize(n_tick);
        for (int64_t i = 0; i < n_tick; ++i) {
            s.times[i] = tick_times[i] * 1000;
            s.codes[i] = tick_codes[i];
            s.locs[i]  = static_cast<uint32_t>(i);
        }
        sequences.push_back(std::move(s));
    }

    // 3b. SH channels (< 1000): merge order+trans by biz_index
    std::set<int32_t> sh_ch_set;
    for (auto& [ch, _] : order_by_ch) if (ch < SH_CHANNEL_THRESHOLD) sh_ch_set.insert(ch);
    for (auto& [ch, _] : trans_by_ch) if (ch < SH_CHANNEL_THRESHOLD) sh_ch_set.insert(ch);

    int64_t n_sh_merged = 0;
    for (int32_t ch : sh_ch_set) {
        auto oit = order_by_ch.find(ch);
        auto tit = trans_by_ch.find(ch);
        const auto& o_idx = (oit != order_by_ch.end()) ? oit->second : std::vector<int64_t>{};
        const auto& t_idx = (tit != trans_by_ch.end()) ? tit->second : std::vector<int64_t>{};

        if (o_idx.empty() && t_idx.empty()) continue;

        // Only one side
        if (o_idx.empty()) {
            SeqData s;
            for (auto i : t_idx) {
                s.times.push_back(trans_times[i] * 1000);
                s.types.push_back(2);
                s.codes.push_back(trans_codes[i]);
                s.locs.push_back(static_cast<uint32_t>(i));
            }
            n_sh_merged += static_cast<int64_t>(t_idx.size());
            sequences.push_back(std::move(s));
            continue;
        }
        if (t_idx.empty()) {
            SeqData s;
            for (auto i : o_idx) {
                s.times.push_back(order_times[i] * 1000);
                s.types.push_back(1);
                s.codes.push_back(order_codes[i]);
                s.locs.push_back(static_cast<uint32_t>(i));
            }
            n_sh_merged += static_cast<int64_t>(o_idx.size());
            sequences.push_back(std::move(s));
            continue;
        }

        // Both sides: two-way merge by biz_index
        std::vector<int64_t> o_biz_v(o_idx.size()), t_biz_v(t_idx.size());
        for (size_t k = 0; k < o_idx.size(); ++k) o_biz_v[k] = order_biz[o_idx[k]];
        for (size_t k = 0; k < t_idx.size(); ++k) t_biz_v[k] = trans_biz[t_idx[k]];

        auto merge_order = two_way_merge_by_biz(
            o_biz_v.data(), static_cast<int64_t>(o_biz_v.size()),
            t_biz_v.data(), static_cast<int64_t>(t_biz_v.size()));

        int64_t no = static_cast<int64_t>(o_idx.size());
        SeqData s;
        s.times.resize(merge_order.size());
        s.types.resize(merge_order.size());
        s.codes.resize(merge_order.size());
        s.locs.resize(merge_order.size());

        for (size_t k = 0; k < merge_order.size(); ++k) {
            uint32_t mo = merge_order[k];
            if (mo < static_cast<uint32_t>(no)) {
                auto oi = o_idx[mo];
                s.times[k] = order_times[oi] * 1000;
                s.types[k] = 1;
                s.codes[k] = order_codes[oi];
                s.locs[k]  = static_cast<uint32_t>(oi);
            } else {
                auto ti = t_idx[mo - no];
                s.times[k] = trans_times[ti] * 1000;
                s.types[k] = 2;
                s.codes[k] = trans_codes[ti];
                s.locs[k]  = static_cast<uint32_t>(ti);
            }
        }
        n_sh_merged += static_cast<int64_t>(merge_order.size());
        sequences.push_back(std::move(s));
    }

    // 3c. SZ channels (>= 1000): separate order and trans
    for (auto& [ch, idx] : order_by_ch) {
        if (ch < SH_CHANNEL_THRESHOLD) continue;
        SeqData s;
        for (auto i : idx) {
            s.times.push_back(order_times[i] * 1000);
            s.types.push_back(1);
            s.codes.push_back(order_codes[i]);
            s.locs.push_back(static_cast<uint32_t>(i));
        }
        sequences.push_back(std::move(s));
    }
    for (auto& [ch, idx] : trans_by_ch) {
        if (ch < SH_CHANNEL_THRESHOLD) continue;
        SeqData s;
        for (auto i : idx) {
            s.times.push_back(trans_times[i] * 1000);
            s.types.push_back(2);
            s.codes.push_back(trans_codes[i]);
            s.locs.push_back(static_cast<uint32_t>(i));
        }
        sequences.push_back(std::move(s));
    }

    int n_seqs = static_cast<int>(sequences.size());
    LOG("  %d sequences (%zu SH merged channels, %ld SH rows)  (%.1fs)",
        n_seqs, sh_ch_set.size(), n_sh_merged, now_sec() - t0);
    for (int i = 0; i < n_seqs; ++i) {
        auto& sq = sequences[i];
        const char* label = "??";
        bool mixed = false;
        if (sq.size() > 0) {
            int8_t t0v = sq.types[0];
            for (int64_t j = 1; j < sq.size(); ++j)
                if (sq.types[j] != t0v) { mixed = true; break; }
            if (mixed) label = "sh_mg";
            else label = (t0v == 0) ? "tick" : (t0v == 1) ? "order" : "trans";
        }
        LOG("    [%2d] %5s  %12ld rows", i, label, sq.size());
    }

    // Free raw-array references (tables still hold memory)
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

    // Free sequence memory
    sequences.clear();
    sequences.shrink_to_fit();
    merge_seqs.clear();

    // ── 6a. Pre-reorder source tables ────────────────────────────────
    LOG("Pre-reordering source tables by merge order ...");
    t0 = now_sec();

    // Single pass to partition locs and positions by source
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
    merge_loc.clear();
    merge_loc.shrink_to_fit();

    // Parallel Take for 3 sources + extract sort_times
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

    // ── 6b. Chunked gather + write ───────────────────────────────────
    fs::create_directories(out_dir);
    std::string out_path = out_dir + "/sorted_" + date_str + ".parquet";

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
    std::vector<int64_t> cursors = {0, 0, 0};
    int lt_idx = schema->GetFieldIndex("local_time");

    // Helper: build one chunk table with local_time filled in
    auto build_chunk = [&](int64_t start) -> std::shared_ptr<arrow::Table> {
        int64_t end = std::min(start + CHUNK_SIZE, total);
        int64_t chunk_len = end - start;

        auto batch_table = build_sorted_batch(
            merge_src.data() + start, chunk_len,
            sources, cursors, struct_names, schema);

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

    // Pipeline: build chunk N+1 while writing chunk N
    std::shared_ptr<arrow::Table> pending_table;
    arrow::Status write_status;
    std::thread write_thread;

    for (int64_t start = 0; start < total; start += CHUNK_SIZE) {
        auto table = build_chunk(start);
        int64_t chunk_len = std::min(CHUNK_SIZE, total - start);

        // Wait for previous write to finish before starting next
        if (write_thread.joinable()) {
            write_thread.join();
            if (!write_status.ok()) {
                LOG("WRITE ERROR: %s", write_status.ToString().c_str());
                return 1;
            }
        }

        // Launch async write for this chunk, then loop back to build next
        pending_table = std::move(table);
        write_status = arrow::Status::OK();
        write_thread = std::thread([&writer, &pending_table, &write_status]() {
            write_status = writer->WriteTable(*pending_table, pending_table->num_rows());
        });

        written += chunk_len;
        ++n_chunks;

        if (n_chunks % 5 == 0) {
            double el = now_sec() - t0;
            double eta = el / written * (total - written);
            LOG("    %12ld / %ld written  (%.0fs elapsed, ~%.0fs left)",
                written, total, el, eta);
        }
    }

    // Wait for last write
    if (write_thread.joinable()) {
        write_thread.join();
        if (!write_status.ok()) {
            LOG("WRITE ERROR: %s", write_status.ToString().c_str());
            return 1;
        }
    }

    ARROW_OK(writer->Close());
    ARROW_OK(outfile->Close());

    double write_sec = now_sec() - t0;
    LOG("  write done in %.1fs  (%.0f rows/s, %d chunks)",
        write_sec, total / write_sec, n_chunks);

    // ── Summary ──────────────────────────────────────────────────────
    double total_sec = now_sec() - ts;
    LOG("============================================================");
    LOG("Total rows written : %ld", written);
    LOG("Output             : %s", out_path.c_str());
    LOG("Elapsed            : %.1fs (%.1f min)", total_sec, total_sec / 60.0);

    // Verify
    LOG("Verifying output ...");
    auto verify_file = unwrap(arrow::io::ReadableFile::Open(out_path), "verify_open");
    std::unique_ptr<parquet::arrow::FileReader> verify_reader;
    ARROW_OK(parquet::arrow::OpenFile(
        verify_file, arrow::default_memory_pool(), &verify_reader));
    auto file_meta = verify_reader->parquet_reader()->metadata();
    LOG("  row groups : %d", file_meta->num_row_groups());
    LOG("  total rows : %ld", file_meta->num_rows());

    return 0;
}
