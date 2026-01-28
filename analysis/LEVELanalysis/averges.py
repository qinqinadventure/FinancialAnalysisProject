import pandas as pd
import numpy as np

# 计算均线
def calculate_moving_averages(stock_data):
    """计算10日、20日、30日均线"""
    stock_data['MA10'] = stock_data['收盘'].rolling(window=10, min_periods=1).mean()
    stock_data['MA20'] = stock_data['收盘'].rolling(window=20, min_periods=1).mean()
    stock_data['MA30'] = stock_data['收盘'].rolling(window=30, min_periods=1).mean()
    return stock_data


# 判断趋势
def judge_trend(stock_data):
    """判断当前位置趋势"""
    latest = stock_data.iloc[-1]

    # 获取最新收盘价和均线值
    close_price = latest['收盘']
    ma10 = latest.get('MA10', np.nan)
    ma20 = latest.get('MA20', np.nan)
    ma30 = latest.get('MA30', np.nan)

    trend_info = {
        '收盘价': close_price,
        'MA10': ma10,
        'MA20': ma20,
        'MA30': ma30
    }

    # 判断均线排列
    if pd.isna(ma10) or pd.isna(ma20) or pd.isna(ma30):
        trend_info['趋势'] = "数据不足，无法判断"
    elif ma10 > ma20 > ma30 and close_price > ma10:
        trend_info['趋势'] = "强势上行"
    elif ma10 > ma20 > ma30:
        trend_info['趋势'] = "上行趋势"
    elif ma10 < ma20 < ma30 and close_price < ma10:
        trend_info['趋势'] = "强势下行"
    elif ma10 < ma20 < ma30:
        trend_info['趋势'] = "下行趋势"
    elif ma10 > ma20 and ma20 > ma30:
        trend_info['趋势'] = "震荡偏多"
    else:
        trend_info['趋势'] = "震荡整理"

    return trend_info


# 判断与均线关系
def judge_ma_relationship(stock_data):
    """判断当前位置与10日/20日/30日均线关系"""
    latest = stock_data.iloc[-1]
    close_price = latest['收盘']

    ma_relations = {}

    for ma_period in [10, 20, 30]:
        ma_value = latest.get(f'MA{ma_period}', np.nan)
        if pd.isna(ma_value):
            relation = "数据不足"
        else:
            diff_percent = (close_price - ma_value) / ma_value * 100
            if diff_percent > 5:
                relation = f"大幅高于{ma_period}日均线({diff_percent:.2f}%)"
            elif diff_percent > 0:
                relation = f"位于{ma_period}日均线上方({diff_percent:.2f}%)"
            elif diff_percent > -5:
                relation = f"位于{ma_period}日均线下方({abs(diff_percent):.2f}%)"
            else:
                relation = f"大幅低于{ma_period}日均线({abs(diff_percent):.2f}%)"

        ma_relations[f'与MA{ma_period}关系'] = relation

    # 判断均线交叉状态
    if all(pd.notna(latest.get(f'MA{ma}', np.nan)) for ma in [10, 20, 30]):
        if latest['MA10'] > latest['MA20'] > latest['MA30']:
            ma_relations['均线排列'] = "多头排列"
        elif latest['MA10'] < latest['MA20'] < latest['MA30']:
            ma_relations['均线排列'] = "空头排列"
        else:
            ma_relations['均线排列'] = "交织排列"

    return ma_relations
