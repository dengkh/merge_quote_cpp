#pragma once
// Minimal .npy reader for structured arrays with known schemas.
//
// Supported schemas:
//   tick:    S9(ticker)@0, <u4(exch_time)@12, <u8(local_time)@16, itemsize=24
//   channel (classic): <i2(channel)@0, <i8(index)@8, <i8(local_time)@16, itemsize=24
//   channel (delta):   <u4(index_det)@0, <u4(local_time_det)@4, itemsize=8
//     Filenames: {order|trans}_{channel}_{startIdx}_{startLT}.npy
//     Reconstruction: cumsum(delta) + start value from filename

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {
namespace fs = std::filesystem;

struct Header {
    int64_t n_records   = 0;
    int64_t itemsize    = 0;
    int64_t data_offset = 0;
};

struct RawFile {
    std::vector<char> buf;
    Header hdr;
    const char* record(int64_t i) const {
        return buf.data() + hdr.data_offset + i * hdr.itemsize;
    }
};

inline RawFile load_raw(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("npy: cannot open " + path);

    fseek(f, 0, SEEK_END);
    int64_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char magic[6];
    if (fread(magic, 1, 6, f) != 6 ||
        magic[0] != '\x93' || std::memcmp(magic + 1, "NUMPY", 5) != 0) {
        fclose(f);
        throw std::runtime_error("npy: bad magic in " + path);
    }

    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);

    uint32_t hdr_len;
    if (major == 1) { uint16_t h; fread(&h, 2, 1, f); hdr_len = h; }
    else            { fread(&hdr_len, 4, 1, f); }

    std::string hdr_str(hdr_len, '\0');
    fread(&hdr_str[0], 1, hdr_len, f);
    fclose(f);

    // Parse 'shape': (N,)
    int64_t n_records = 0;
    auto sp = hdr_str.find("'shape'");
    if (sp == std::string::npos) sp = hdr_str.find("\"shape\"");
    if (sp != std::string::npos) {
        auto lp = hdr_str.find('(', sp);
        if (lp != std::string::npos) {
            n_records = std::stoll(hdr_str.substr(lp + 1));
        }
    }
    if (n_records <= 0)
        throw std::runtime_error("npy: failed to parse shape in " + path);

    int64_t data_off = 6 + 2 + (major == 1 ? 2 : 4) + hdr_len;
    int64_t itemsize = (file_size - data_off) / n_records;

    RawFile out;
    out.hdr = {n_records, itemsize, data_off};
    out.buf.resize(file_size);
    FILE* f2 = fopen(path.c_str(), "rb");
    fread(out.buf.data(), 1, file_size, f2);
    fclose(f2);
    return out;
}

// ── Tick snapshot NPY ─────────────────────────────────────────────────

struct TickData {
    int64_t n = 0;
    std::vector<int64_t>  ticker;     // S9 → int64
    std::vector<uint32_t> exch_time;
    std::vector<int64_t>  local_time; // ×1000 → ns
};

inline int64_t parse_ticker_s9(const char* s) {
    char buf[10] = {};
    std::memcpy(buf, s, 9);
    return std::atoll(buf);
}

inline TickData load_tick(const std::string& path) {
    auto raw = load_raw(path);
    if (raw.hdr.itemsize != 24)
        throw std::runtime_error("tick npy: itemsize=" +
                                 std::to_string(raw.hdr.itemsize) + " != 24");
    int64_t n = raw.hdr.n_records;
    TickData d;
    d.n = n;
    d.ticker.resize(n);
    d.exch_time.resize(n);
    d.local_time.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        const char* r = raw.record(i);
        d.ticker[i]    = parse_ticker_s9(r);
        uint32_t et; std::memcpy(&et, r + 12, 4);
        d.exch_time[i] = et;
        uint64_t lt; std::memcpy(&lt, r + 16, 8);
        d.local_time[i] = static_cast<int64_t>(lt) * 1000;
    }
    return d;
}

// ── Channel NPY (trans / order) ───────────────────────────────────────

struct ChannelData {
    int64_t n = 0;
    std::vector<int16_t> channel;
    std::vector<int64_t> index;      // trade_index or order_index
    std::vector<int64_t> local_time; // ×1000 → ns
};

inline ChannelData load_channel_file(const std::string& path) {
    auto raw = load_raw(path);
    if (raw.hdr.itemsize != 24)
        throw std::runtime_error("channel npy: itemsize=" +
                                 std::to_string(raw.hdr.itemsize) + " != 24");
    int64_t n = raw.hdr.n_records;
    ChannelData d;
    d.n = n;
    d.channel.resize(n);
    d.index.resize(n);
    d.local_time.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        const char* r = raw.record(i);
        int16_t ch; std::memcpy(&ch, r, 2);
        d.channel[i] = ch;
        int64_t idx; std::memcpy(&idx, r + 8, 8);
        d.index[i] = idx;
        int64_t lt;  std::memcpy(&lt, r + 16, 8);
        d.local_time[i] = lt * 1000;
    }
    return d;
}

inline ChannelData load_channel_dir(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(dir)) {
        auto fn = e.path().filename().string();
        if (e.path().extension() == ".npy" && fn.rfind("channel-", 0) == 0)
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    ChannelData merged;
    for (auto& fp : files) {
        auto chunk = load_channel_file(fp);
        if (chunk.n == 0) continue;
        merged.channel.insert(merged.channel.end(),
                              chunk.channel.begin(), chunk.channel.end());
        merged.index.insert(merged.index.end(),
                            chunk.index.begin(), chunk.index.end());
        merged.local_time.insert(merged.local_time.end(),
                                 chunk.local_time.begin(), chunk.local_time.end());
        merged.n += chunk.n;
    }
    return merged;
}

// ── Delta-encoded NPY (order/trans) ──────────────────────────────────
// Filename: {order|trans}_{channel}_{startIdx}_{startLocalTime}.npy
// dtype: [('*_index_det', '<u4'), ('local_time_det', '<u4')], itemsize=8
// Reconstruction: cumsum(deltas) + start values extracted from filename.

struct DeltaFileInfo {
    std::string path;
    int16_t     channel;
    int64_t     start_idx;
    int64_t     start_lt;
};

inline bool parse_delta_filename(const std::string& stem,
                                 const std::string& prefix,
                                 DeltaFileInfo& out)
{
    // Expected: {prefix}_{channel}_{startIdx}_{startLT}
    if (stem.rfind(prefix + "_", 0) != 0) return false;
    auto rest = stem.substr(prefix.size() + 1); // channel_startIdx_startLT

    auto p1 = rest.find('_');
    if (p1 == std::string::npos) return false;
    auto p2 = rest.find('_', p1 + 1);
    if (p2 == std::string::npos) return false;

    try {
        out.channel  = static_cast<int16_t>(std::stoi(rest.substr(0, p1)));
        out.start_idx = std::stoll(rest.substr(p1 + 1, p2 - p1 - 1));
        out.start_lt  = std::stoll(rest.substr(p2 + 1));
    } catch (...) {
        return false;
    }
    return true;
}

inline ChannelData load_delta_channel_file(const DeltaFileInfo& info) {
    auto raw = load_raw(info.path);
    if (raw.hdr.itemsize != 8)
        throw std::runtime_error("delta npy: itemsize=" +
                                 std::to_string(raw.hdr.itemsize) + " != 8 in " + info.path);
    int64_t n = raw.hdr.n_records;
    ChannelData d;
    d.n = n;
    d.channel.assign(n, info.channel);
    d.index.resize(n);
    d.local_time.resize(n);

    // Prefix-sum reconstruction from uint32 deltas
    int64_t cum_idx = info.start_idx;
    int64_t cum_lt  = info.start_lt;

    for (int64_t i = 0; i < n; ++i) {
        const char* r = raw.record(i);
        uint32_t d_idx, d_lt;
        std::memcpy(&d_idx, r, 4);
        std::memcpy(&d_lt,  r + 4, 4);
        cum_idx += d_idx;
        cum_lt  += d_lt;
        d.index[i]      = cum_idx;
        d.local_time[i]  = cum_lt * 1000;  // μs → ns
    }
    return d;
}

inline ChannelData load_delta_channel_dir(const std::string& dir,
                                          const std::string& type_prefix)
{
    std::vector<DeltaFileInfo> infos;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".npy") continue;
        DeltaFileInfo fi;
        fi.path = e.path().string();
        if (parse_delta_filename(e.path().stem().string(), type_prefix, fi))
            infos.push_back(std::move(fi));
    }
    std::sort(infos.begin(), infos.end(),
              [](const DeltaFileInfo& a, const DeltaFileInfo& b) {
                  if (a.channel != b.channel) return a.channel < b.channel;
                  return a.start_idx < b.start_idx;
              });

    ChannelData merged;
    for (auto& fi : infos) {
        auto chunk = load_delta_channel_file(fi);
        if (chunk.n == 0) continue;
        merged.channel.insert(merged.channel.end(),
                              chunk.channel.begin(), chunk.channel.end());
        merged.index.insert(merged.index.end(),
                            chunk.index.begin(), chunk.index.end());
        merged.local_time.insert(merged.local_time.end(),
                                 chunk.local_time.begin(), chunk.local_time.end());
        merged.n += chunk.n;
    }
    return merged;
}

} // namespace npy
