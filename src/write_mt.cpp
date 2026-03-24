// write_mt.cpp — Minimal C++ bridge for multi-threaded parquet writing.
// Compiled as libwrite_mt.so and called from Python via ctypes.
// Data is passed zero-copy through the Arrow C Data Interface.

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <cstdint>
#include <memory>
#include <string>

struct WriterState {
    std::unique_ptr<parquet::arrow::FileWriter> writer;
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    std::shared_ptr<arrow::Schema> schema;
};

extern "C" {

// Open a parquet writer with multi-threaded compression.
// schema_addr: pointer to an exported ArrowSchema (consumed).
// Returns opaque WriterState*, or nullptr on error.
void* writer_open(uintptr_t schema_addr, const char* path,
                  const char* compression, int compression_level) {
    try {
        auto c_schema = reinterpret_cast<struct ArrowSchema*>(schema_addr);
        auto schema = arrow::ImportSchema(c_schema).ValueOrDie();

        auto outfile = arrow::io::FileOutputStream::Open(std::string(path)).ValueOrDie();

        arrow::Compression::type comp = arrow::Compression::ZSTD;
        std::string comp_str(compression);
        if (comp_str == "snappy")       comp = arrow::Compression::SNAPPY;
        else if (comp_str == "lz4")     comp = arrow::Compression::LZ4;
        else if (comp_str == "gzip")    comp = arrow::Compression::GZIP;
        else if (comp_str == "brotli")  comp = arrow::Compression::BROTLI;
        else if (comp_str == "none")    comp = arrow::Compression::UNCOMPRESSED;

        auto props_builder = parquet::WriterProperties::Builder();
        props_builder.compression(comp);
        if (compression_level > 0)
            props_builder.compression_level(compression_level);
        auto wr_props = props_builder.build();

        auto ar_props = parquet::ArrowWriterProperties::Builder()
                            .set_use_threads(true)
                            ->build();

        auto writer = parquet::arrow::FileWriter::Open(
            *schema, arrow::default_memory_pool(), outfile, wr_props, ar_props
        ).ValueOrDie();

        return new WriterState{std::move(writer), std::move(outfile), std::move(schema)};
    } catch (const std::exception& e) {
        fprintf(stderr, "writer_open error: %s\n", e.what());
        return nullptr;
    }
}

// Write one RecordBatch (exported as ArrowArray, consumed).
// Returns 0 on success, -1 on error.
int writer_write_batch(void* state_ptr, uintptr_t array_addr) {
    try {
        auto* state = static_cast<WriterState*>(state_ptr);
        auto c_array = reinterpret_cast<struct ArrowArray*>(array_addr);
        auto batch = arrow::ImportRecordBatch(c_array, state->schema).ValueOrDie();
        auto status = state->writer->WriteRecordBatch(*batch);
        if (!status.ok()) {
            fprintf(stderr, "WriteRecordBatch: %s\n", status.ToString().c_str());
        }
        return status.ok() ? 0 : -1;
    } catch (const std::exception& e) {
        fprintf(stderr, "writer_write_batch error: %s\n", e.what());
        return -1;
    }
}

// Close writer and free resources. Returns 0 on success.
int writer_close(void* state_ptr) {
    try {
        auto* state = static_cast<WriterState*>(state_ptr);
        auto s1 = state->writer->Close();
        auto s2 = state->outfile->Close();
        delete state;
        return (s1.ok() && s2.ok()) ? 0 : -1;
    } catch (const std::exception& e) {
        fprintf(stderr, "writer_close error: %s\n", e.what());
        delete static_cast<WriterState*>(state_ptr);
        return -1;
    }
}

}  // extern "C"
