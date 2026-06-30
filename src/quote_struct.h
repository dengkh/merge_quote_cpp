#pragma once

#include <stdint.h>

#pragma pack(push)
#pragma pack(8)

// 股票数据，类型1
typedef struct {
    char wind_code[32];        // 600001.SH
    char ticker[32];           // 原始Code
    int action_day;            // 业务发生日(自然日)
    int trading_day;           // 交易日
    int exch_time;             // 时间(HHMMSSmmm)
    int status;                // 股票状态(见feed_status)
    uint32_t pre_close_px;     // 前收盘价 * 10000
    uint32_t open_px;          // 开盘价 * 10000
    uint32_t high_px;          // 最高价 * 10000
    uint32_t low_px;           // 最低价 * 10000
    uint32_t last_px;          // 最新价 * 10000
    uint32_t ap_array[10];     // 申卖价 * 10000
    uint32_t av_array[10];     // 申卖量
    uint32_t bp_array[10];     // 申买价 * 10000
    uint32_t bv_array[10];     // 申买量
    uint32_t num_of_trades;    // 成交笔数
    int64_t total_vol;         // 成交总量
    int64_t total_notional;    // 成交总额准确值,Turnover
    int64_t total_bid_vol;     // 委托买入总量
    int64_t total_ask_vol;     // 委托卖出总量
    uint32_t weighted_avg_bp;  // 加权平均委买价格 * 10000
    uint32_t weighted_avg_ap;  // 加权平均委卖价格 * 10000
    int IOPV;                  // IOPV净值估值  （基金） * 10000
    int yield_to_maturity;     // 到期收益率    （债券） * 10000
    uint32_t upper_limit_px;   // 涨停价 * 10000
    uint32_t lower_limit_px;   // 跌停价 * 10000
    char prefix[4];            // 证券信息前缀
    int PE1;                   // 市盈率1   未使用（当前值为0）
    int PE2;                   // 市盈率2   未使用（当前值为0）
    int change;  // 升跌2（对比上一笔）   未使用（当前值为0）
} Stock_Tick;

typedef struct {
    int serial;
    int mi_type;
    uint64_t local_time;
    uint64_t exchange_time;
    Stock_Tick quote;
} Stock_Tick_Series;

// 逐笔成交，类型2
typedef struct {
    uint16_t channel;
    char symbol[9];
    uint8_t market;
    int int_time;  // 成交时间
    int trade_price;
    char bsflag;
    char trade_type;
    int64_t trade_index;
    int64_t trade_volume;  // 成交数量
    int64_t trade_amount;  // 成交金额
    int64_t sell_id;
    int64_t buy_id;
    int64_t biz_index;
} Stock_Transaction;

typedef struct {
    int serial;
    int mi_type;
    uint64_t local_time;
    uint64_t exchange_time;
    Stock_Transaction quote;
} Stock_Transaction_Series;

// 逐笔委托，类型3
typedef struct {
    char order_type;
    uint8_t market;
    char bsflag;
    char symbol[9];
    int int_time;  // 委托时间
    int64_t order_index;
    int64_t order_volume;  // 委托数量
    uint64_t orderorino;
    uint64_t biz_index;
    int order_price;  // 委托价格
    uint16_t channel;
} Stock_Order;
  
typedef struct {
    int serial;
    int mi_type;
    uint64_t local_time;
    uint64_t exchange_time;
    Stock_Order quote;
} Stock_Order_Series;
  
#pragma pack(pop)

