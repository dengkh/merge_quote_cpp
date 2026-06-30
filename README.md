# parquet_sort — 行情数据合并排序工具

将 tick / trans / order 三类行情 Parquet 数据按时间合并排序，并与 NPY local_time 数据进行关联，输出按 `local_time` 严格递增的每类型 Parquet 文件。

## 构建

```bash
cd /mnt/data/private/dev002/code/merge_quote_cpp
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

产出可执行文件：`build/parquet_sort`

## 命令格式

```
parquet_sort <YYYYMMDD> <data_dir> [options]
```

### 位置参数

| 参数 | 说明 |
|------|------|
| `YYYYMMDD` | 交易日期，8位数字格式 |
| `data_dir` | 输入 Parquet 数据目录，包含 `1/`（tick）、`2/`（trans）、`3/`（order）子目录，每个子目录下有按股票分割的 `.parquet` 文件 |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--npy-dir DIR` | 路径 | （空） | NPY local_time 数据根目录。**指定后启用 Stage 2**（NPY 合并 + 按类型输出）；未指定则仅执行 Stage 1 |
| `--output-base DIR` | 路径 | `./output/stg2` | Stage 2 输出根目录。输出路径为 `{output-base}/{YYYY.MM}/{YYYY.MM.DD}/default/{1,2,3}/all.parquet` |
| `--output-dir DIR` | 路径 | `./output` | Stage 1 中间文件输出目录（`sorted_{date}.parquet`）|
| `--write-intermediate` | 开关 | 关 | 是否输出 Stage 1 中间 sorted parquet 文件（调试用）|
| `--delta-npy` | 开关 | 关 | 使用 delta 编码 NPY 格式（默认使用 classic 格式）|
| `--post-process` | 开关 | 关 | Stage 2 完成后执行后处理：将 `all.parquet` 按股票拆分为单独文件 |
| `--log-file FILE` | 路径 | （空） | 将日志同时写入指定文件（默认仅输出到 stderr）|

### 行为规则

- **未指定 `--npy-dir`**：仅执行 Stage 1，自动开启 `--write-intermediate` 输出中间文件
- **指定 `--npy-dir`**：执行 Stage 1 + Stage 2（NPY 合并 + 按类型输出 `all.parquet`）

## 处理流程

### Stage 1: K-way 合并排序

1. **读取数据**：用 16 线程共享线程池并行读取 `data_dir` 下的 tick/trans/order parquet 文件
2. **预排序**：
   - tick：按 `(exchange_time, ticker)` 排序
   - order：按 `(channel, biz_index, order_index)` 排序
   - trans：按 `(channel, biz_index, trade_index)` 排序
3. **构建子序列**：每个 channel 的 order + trans 按 merge key 合并为一个序列
   - **沪市**（channel < 1000）：merge key = `biz_index`（共享序列号）
   - **深市**（channel >= 1000）：merge key = `order_index`/`trade_index`（applseq，共享序列空间）
4. **K-way merge**：用 min-heap 按 `(sort_time, type_priority, stock_code)` 合并所有序列
   - type_priority：tick=0 < order=1 < trans=2
5. **计算 local_time**：在合并流上生成全局严格递增的 `local_time`（基于 exchange_time，冲突 +1ns）

### Stage 2: NPY 合并 + 按类型输出

6. **分割 front/back**：以当日 09:40:00 CST 为界
   - front（09:15 ~ 09:40）：NPY 提供 local_time
   - back（09:40 之后）：使用 Stage 1 的 local_time + offset
7. **加载 NPY 数据**
8. **Front section join**：
   - tick：right-join on `(ticker, exch_time)` — NPY 行驱动输出
   - trans：left-join on `(channel, trade_index)` — parquet 行保留，匹配 NPY 获取 local_time
   - order：left-join on `(channel, order_index)` — 同上
9. **Front K-way re-merge**：和 Stage 1 相同的序列构建 + K-way merge，按 NPY local_time 排序
10. **计算 offset**：确保 front 最后一行 local_time < back 第一行 local_time
11. **按类型输出**：分别写入 tick/trans/order 的 `all.parquet`
    - front section：使用 NPY 合并后的 local_time
    - back section：`local_time = Stage1_local_time + offset`
    - serial 列：从 0 开始顺序编号
12. **后处理**（可选）：按股票拆分为独立 parquet 文件

## NPY 输入格式

### Classic 格式（默认）

`--npy-dir` 指向根目录，程序按以下路径查找：

```
{npy-dir}/{YYYYMMDD}/snapshot.npy              ← tick
{npy-dir}/{YYYYMMDD}/trans_localtime_by_channel/channel-{N}.npy  ← trans
{npy-dir}/{YYYYMMDD}/order_localtime_by_channel/channel-{N}.npy  ← order
```

典型路径：`/mnt/data/public/quant001/local_times/stocks/91500000-94000000`

### Delta 格式（`--delta-npy`）

`--npy-dir` 直接指向包含 NPY 文件的目录：

```
{npy-dir}/snapshot.npy                         ← tick
{npy-dir}/trans_{channel}_{baseIdx}_{baseTs}.npy   ← trans (delta 编码)
{npy-dir}/order_{channel}_{baseIdx}_{baseTs}.npy   ← order (delta 编码)
```

Delta 文件中 `index` 和 `local_time` 以差分（delta）形式存储，程序读取后做前缀和还原。

## 输出格式

### 目录结构

```
{output-base}/
  {YYYY.MM}/
    {YYYY.MM.DD}/
      default/
        1/all.parquet    ← tick (StockSnapshotParquet schema)
        2/all.parquet    ← trans (StockTransactionParquet schema)
        3/all.parquet    ← order (StockOrderParquet schema)
```

### 输出特性

- 每个 `all.parquet` 内 `local_time` **全局严格递增**
- 三个类型的 `local_time` **互不重叠且全局严格递增**
- Parquet 压缩：ZSTD level 3，row group 大小 20,000,000 行
- `serial` 列：从 0 开始的连续整数
- 沪市 channel 内 order + trans 按 `biz_index` 交织排序
- 深市 channel 内 order + trans 按 `applseq` 交织排序

## 使用示例

### 单日执行

```bash
./build/parquet_sort 20230103 \
    /mnt/data/data/ParquetDataV2/2023.01/2023.01.03/default \
    --npy-dir /mnt/data/public/quant001/local_times/stocks/91500000-94000000 \
    --output-base /mnt/beegfs/quant002/hds_work/CppGen
```

### 单日执行（带日志文件）

```bash
./build/parquet_sort 20230103 \
    /mnt/data/data/ParquetDataV2/2023.01/2023.01.03/default \
    --npy-dir /mnt/data/public/quant001/local_times/stocks/91500000-94000000 \
    --output-base /mnt/beegfs/quant002/hds_work/CppGen \
    --log-file /tmp/parquet_sort_20230103.log
```

### 仅 Stage 1（输出中间文件）

```bash
./build/parquet_sort 20230103 \
    /mnt/data/data/ParquetDataV2/2023.01/2023.01.03/default \
    --output-dir ./output
# 输出: ./output/sorted_20230103.parquet
```

### 批量执行（run_parquet.py）

```bash
python3 run_parquet.py <start_date> <end_date> <data_base> <npy_dir> <output_base> [extra_flags...]
```

```bash
# 2023全年
python3 run_parquet.py 2023-01-01 2023-12-22 \
    /mnt/data/data/ParquetDataV2 \
    /mnt/data/public/quant001/local_times/stocks/91500000-94000000 \
    /mnt/beegfs/quant002/hds_work/CppGen

# 带后处理
python3 run_parquet.py 2023-01-01 2023-01-31 \
    /mnt/data/data/ParquetDataV2 \
    /mnt/data/public/quant001/local_times/stocks/91500000-94000000 \
    /mnt/beegfs/quant002/hds_work/CppGen \
    --post-process
```

`run_parquet.py` 会自动获取时间范围内的所有 A 股交易日（基于 `exchange_calendars` 上交所日历），逐日调用 `parquet_sort`。

## 输出验证（verify_npy_output.py）

```bash
python3 verify_npy_output.py <YYYYMMDD> --output-base <output_base> [options]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `YYYYMMDD` | — | 验证日期 |
| `--output-base` | `/mnt/data/data/CppGen` | 输出根目录 |
| `--sorted-dir` | — | sorted 中间文件目录（可选，用于 check 3/4）|
| `--npy-format` | `classic` | NPY 格式：`classic` 或 `delta` |
| `--npy-base` | 自动选择 | NPY 根目录 |
| `--types` | `tick trans order` | 要验证的类型 |

### 验证项

| # | 检查项 | 范围 | PASS 条件 |
|---|--------|------|-----------|
| 1 | local_time 严格递增 | 每个类型 | 无违反，无 NULL |
| 2 | NPY join 匹配 | 每个类型 | tick: 匹配数 > 0；trans/order: 匹配且排序保持 |
| 3 | 行数完整性 | 每个类型 | 需要 sorted 中间文件，无则 SKIP |
| 4 | 业务列完整性 | trans/order | 需要 sorted 中间文件，无则 SKIP |
| 5 | 通道内 index 递增 | trans/order | back section 每 channel 的 index 非递减 |
| 6 | 沪市 biz_index 排序 | 跨类型 | back section 每 SH channel biz_index 非递减 |
| 7 | 深市 applseq 排序 | 跨类型 | back section 每 SZ channel applseq 非递减 |
| 8 | 全局 local_time 递增 | 跨类型 | 三类型合并后 local_time 严格递增 |

### 验证示例

```bash
python3 verify_npy_output.py 20230103 \
    --output-base /mnt/beegfs/quant002/hds_work/CppGen
```
