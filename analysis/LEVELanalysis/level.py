import analysis.LEVELanalysis.averges as avg

# 判断压力位
def judge_resistance_levels(stock_data, lookback_period=60):
    """判断当前位置压力位"""
    recent_data = stock_data.tail(lookback_period)

    resistance_levels = {
        '近期高点': recent_data['最高'].max(),
        '近期收盘高点': recent_data['收盘'].max(),
        '成交密集区上沿': recent_data['收盘'].quantile(0.8),  # 80%分位数
        '近期跳空缺口': None
    }

    # 查找跳空缺口（向下跳空）
    recent_data = recent_data.sort_values('日期')
    for i in range(1, len(recent_data)):
        if recent_data.iloc[i]['最低'] > recent_data.iloc[i - 1]['最高']:
            resistance_levels['近期跳空缺口'] = recent_data.iloc[i - 1]['最高']
            break

    return resistance_levels

# 判断当前位置
def getLevel(stock_data):
    """
    综合判断股票当前位置
    参数:
        stock_data: DataFrame，包含日期、开盘、收盘、最高、最低等列
    返回:
        dict: 包含趋势、压力位、均线关系的综合信息
    """
    # 确保数据按日期排序
    if '日期' in stock_data.columns:
        stock_data = stock_data.sort_values('日期').copy()

    # 第一步：计算均线
    stock_data = avg.calculate_moving_averages(stock_data)

    # 第二步：判断趋势
    trend_result = avg.judge_trend(stock_data)

    # 第三步：判断压力位
    resistance_result = judge_resistance_levels(stock_data)

    # 第四步：判断与均线关系
    ma_result = avg.judge_ma_relationship(stock_data)

    # 综合结果
    comprehensive_result = {
        '趋势分析': trend_result,
        '压力位分析': resistance_result,
        '均线关系分析': ma_result,
        '分析时间': stock_data.iloc[-1]['日期'] if '日期' in stock_data.columns else '未知'
    }

    return comprehensive_result