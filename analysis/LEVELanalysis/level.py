import analysis.LEVELanalysis.averges as avg
import pandas as pd
import numpy as np
from scipy import stats

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


def judge_historical_high_low(stock_data, period_years=3):
    """
    判断股票是否处于历史高位或低位
    通过对比当前价格与历史价格的相对位置来分析[1,2](@ref)
    """
    if len(stock_data) < 250 * period_years:
        period_years = len(stock_data) // 250
        if period_years == 0:
            return {'历史位置分析': '数据不足，无法分析'}

    lookback_days = 250 * period_years
    historical_data = stock_data.tail(lookback_days)

    current_price = historical_data.iloc[-1]['收盘']
    historical_high = historical_data['最高'].max()
    historical_low = historical_data['最低'].min()

    # 计算当前位置百分比
    position_ratio = (current_price - historical_low) / (historical_high - historical_low) * 100

    # 判断高低位区域[3](@ref)
    if position_ratio >= 80:
        position_level = "历史高位"
    elif position_ratio >= 60:
        position_level = "相对高位"
    elif position_ratio >= 40:
        position_level = "中间位置"
    elif position_ratio >= 20:
        position_level = "相对低位"
    else:
        position_level = "历史低位"

    return {
        '当前价格': current_price,
        '历史高点': historical_high,
        '历史低点': historical_low,
        '位置百分比': round(position_ratio, 2),
        '位置级别': position_level,
        '分析周期': f"{period_years}年"
    }


def calculate_valuation_quantiles(valuation_data, current_pe, current_pb):
    """
    使用五分位法判断估值区域[6,7](@ref)
    五线谱估值法：极高估(95%)、高估(75%)、合理(50%)、低估(25%)、极低估(5%)
    """

    def get_quantile_value(data, quantile):
        return np.percentile(data, quantile * 100)

    pe_quantiles = {
        '极低估': get_quantile_value(valuation_data['pe'], 0.05),
        '低估': get_quantile_value(valuation_data['pe'], 0.25),
        '合理': get_quantile_value(valuation_data['pe'], 0.50),
        '高估': get_quantile_value(valuation_data['pe'], 0.75),
        '极高估': get_quantile_value(valuation_data['pe'], 0.95)
    }

    pb_quantiles = {
        '极低估': get_quantile_value(valuation_data['pb'], 0.05),
        '低估': get_quantile_value(valuation_data['pb'], 0.25),
        '合理': get_quantile_value(valuation_data['pb'], 0.50),
        '高估': get_quantile_value(valuation_data['pb'], 0.75),
        '极高估': get_quantile_value(valuation_data['pb'], 0.95)
    }

    # 判断当前PE/PB所在分位[8](@ref)
    def get_valuation_level(current_value, quantiles):
        if current_value <= quantiles['极低估']:
            return '极低估'
        elif current_value <= quantiles['低估']:
            return '低估'
        elif current_value <= quantiles['合理']:
            return '合理'
        elif current_value <= quantiles['高估']:
            return '高估'
        else:
            return '极高估'

    pe_level = get_valuation_level(current_pe, pe_quantiles)
    pb_level = get_valuation_level(current_pb, pb_quantiles)

    # 综合估值判断[5](@ref)
    if pe_level in ['极低估', '低估'] and pb_level in ['极低估', '低估']:
        overall_level = '低估区域'
    elif pe_level in ['高估', '极高估'] and pb_level in ['高估', '极高估']:
        overall_level = '高估区域'
    else:
        overall_level = '合理区域'

    return {
        'PE分位': pe_level,
        'PB分位': pb_level,
        '综合估值': overall_level,
        'PE详细分位': pe_quantiles,
        'PB详细分位': pb_quantiles,
        '当前PE': current_pe,
        '当前PB': current_pb
    }


def getLevel(stock_data, valuation_data=None, current_pe=None, current_pb=None):
    """
    综合判断股票当前位置（优化版）
    参数:
        stock_data: DataFrame，包含日期、开盘、收盘、最高、最低等列
        valuation_data: DataFrame，包含历史PE/PB数据
        current_pe: 当前市盈率
        current_pb: 当前市净率
    返回:
        dict: 包含趋势、压力位、历史位置、估值区域的综合信息
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

    # 第五步：判断历史高低位[1,2](@ref)
    historical_result = judge_historical_high_low(stock_data)

    # 第六步：估值分析（如果有估值数据）[6,7](@ref)
    valuation_result = None
    if valuation_data is not None and current_pe is not None and current_pb is not None:
        valuation_result = calculate_valuation_quantiles(valuation_data, current_pe, current_pb)
    elif current_pe is not None and current_pb is not None:
        # 如果没有历史估值数据，使用简单规则判断[5](@ref)
        valuation_result = {
            'PE水平': '低估值' if current_pe < 15 else '高估值' if current_pe > 25 else '合理估值',
            'PB水平': '低估值' if current_pb < 1 else '高估值' if current_pb > 2.5 else '合理估值',
            '当前PE': current_pe,
            '当前PB': current_pb
        }

    # 综合结果
    comprehensive_result = {
        '趋势分析': trend_result,
        '压力位分析': resistance_result,
        '均线关系分析': ma_result,
        '历史位置分析': historical_result,
        '估值分析': valuation_result,
        '分析时间': stock_data.iloc[-1]['日期'] if '日期' in stock_data.columns else '未知'
    }

    # 生成投资建议[3](@ref)
    comprehensive_result['综合建议'] = generate_investment_suggestion(comprehensive_result)

    return comprehensive_result


def generate_investment_suggestion(analysis_result):
    """
    根据综合分析结果生成投资建议[3,5](@ref)
    """
    historical_position = analysis_result['历史位置分析'].get('位置级别', '')
    valuation = analysis_result['估值分析']

    if isinstance(valuation, dict):
        if '综合估值' in valuation:
            valuation_level = valuation['综合估值']
        else:
            valuation_level = f"{valuation.get('PE水平', '')}+{valuation.get('PB水平', '')}"
    else:
        valuation_level = "未知"

    trend = analysis_result['趋势分析']['趋势']

    # 综合判断逻辑
    if '低位' in historical_position and '低估' in str(valuation_level) and '上升' in trend:
        return "强烈关注：历史低位+估值低估+趋势向上"
    elif '高位' in historical_position and '高估' in str(valuation_level) and '下降' in trend:
        return "风险提示：历史高位+估值高估+趋势向下"
    elif '低位' in historical_position or '低估' in str(valuation_level):
        return "关注机会：处于相对低位或低估区域"
    elif '高位' in historical_position or '高估' in str(valuation_level):
        return "保持谨慎：处于相对高位或高估区域"
    else:
        return "中性观望：处于合理区间"