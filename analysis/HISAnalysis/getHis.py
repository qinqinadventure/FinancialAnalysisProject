import numpy as np
import analysis.PCAanalysis.directParams as DP
import pandas as pd


def getHisAnalysis(history_data, quantity_data, analysis_target):
    """
    历史分位分析函数 - 支持analysis_target为列表，支持复合表达式
    参数:
        history_data: 历史数据DataFrame
        quantity_data: 近期数据DataFrame
        analysis_target: 分析目标列名，可以是字符串或字符串列表，支持简单列名和复合表达式

    返回:
        dict: 包含分位分析和排名信息的字典
    """
    results = {}

    # 确保analysis_target是列表格式
    if isinstance(analysis_target, str):
        analysis_target = [analysis_target]

    # 获取所有可用列名[1,3](@ref)
    available_columns = list(history_data.columns)
    print(f"可用列名: {available_columns}")

    for target in analysis_target:
        target_results = {}

        try:
            # 检查是否为复合表达式（包含运算符）
            if any(op in target for op in ['+', '-', '*', '/']):
                # 处理复合表达式
                target_results = _process_compound_expression(history_data, quantity_data, target, available_columns)
            else:
                # 处理简单列名
                target_results = _process_simple_column(history_data, quantity_data, target, available_columns)

        except Exception as e:
            target_results = {
                '分析目标': target,
                '错误信息': f"处理失败: {str(e)}",
                '可用列名': available_columns
            }

        results[target] = target_results

    # 添加总体信息
    results['总体信息'] = {
        '分析目标数量': len(analysis_target),
        '分析目标列表': analysis_target,
        '成功分析数量': sum(1 for target in analysis_target if '错误信息' not in results[target]),
        '历史数据形状': history_data.shape,
        '近期数据形状': quantity_data.shape
    }

    return results


def _process_simple_column(history_data, quantity_data, column_name, available_columns):
    """处理简单列名分析"""
    # 验证列名是否存在[1,2](@ref)
    if column_name not in available_columns:
        raise KeyError(f"列名 '{column_name}' 不存在。可用列名: {available_columns}")

    result = {}

    # 获取当前值
    current_value = quantity_data[column_name].iloc[-1]

    # 历史分位分析
    history_values = history_data[column_name].dropna()
    if len(history_values) == 0:
        raise ValueError(f"列 '{column_name}' 的历史数据为空")

    percentile_rank = np.sum(history_values <= current_value) / len(history_values) * 100

    # 填充结果
    result.update(_calculate_basic_metrics(column_name, current_value, percentile_rank))
    result.update(_calculate_recent_metrics(quantity_data, column_name, current_value))
    result['数据信息'] = {
        '历史数据量': len(history_values),
        '近期数据量': len(quantity_data[column_name].dropna())
    }

    return result


def _process_compound_expression(history_data, quantity_data, expression, available_columns):
    """处理复合表达式分析"""
    # 解析表达式中的列名
    import re
    # 提取表达式中可能的列名（中文字符连续出现）
    potential_columns = re.findall(r'[\u4e00-\u9fa5]+', expression)

    # 验证列名是否存在[3](@ref)
    valid_columns = []
    for col in potential_columns:
        if col in available_columns:
            valid_columns.append(col)

    if len(valid_columns) < 2:
        raise ValueError(f"表达式中需要至少两个有效列名，找到的列: {valid_columns}。可用列名: {available_columns}")

    result = {}

    try:
        # 为历史数据和近期数据计算表达式结果
        history_data_copy = history_data.copy()
        quantity_data_copy = quantity_data.copy()

        # 安全地执行表达式
        history_data_copy['复合指标'] = history_data_copy.eval(expression, engine='python')
        quantity_data_copy['复合指标'] = quantity_data_copy.eval(expression, engine='python')

        # 获取当前值
        current_value = quantity_data_copy['复合指标'].iloc[-1]

        # 历史分位分析
        history_values = history_data_copy['复合指标'].dropna()
        if len(history_values) == 0:
            raise ValueError(f"复合表达式 '{expression}' 的历史数据为空")

        percentile_rank = np.sum(history_values <= current_value) / len(history_values) * 100

        # 填充结果
        result.update(_calculate_basic_metrics(expression, current_value, percentile_rank))
        result.update(_calculate_recent_metrics(quantity_data_copy, '复合指标', current_value))
        result['数据信息'] = {
            '历史数据量': len(history_values),
            '近期数据量': len(quantity_data_copy['复合指标'].dropna()),
            '使用列名': valid_columns
        }

    except Exception as e:
        raise ValueError(f"计算表达式 '{expression}' 处理失败: {str(e)}")

    return result


def _calculate_basic_metrics(target_name, current_value, percentile_rank):
    """计算基本指标"""
    metrics = {
        '分析目标': target_name,
        '当前值': float(current_value),
        '历史分位': f"{percentile_rank:.2f}%",
        '高于历史比例': f"{percentile_rank:.2f}%",
        '低于历史比例': f"{(100 - percentile_rank):.2f}%"
    }

    # 判断历史水平
    if percentile_rank >= 90:
        metrics['历史水平'] = "极高水平(前10%)"
    elif percentile_rank >= 70:
        metrics['历史水平'] = "高水平(前30%)"
    elif percentile_rank >= 30:
        metrics['历史水平'] = "中等水平"
    elif percentile_rank >= 10:
        metrics['历史水平'] = "低水平(后30%)"
    else:
        metrics['历史水平'] = "极低水平(后10%)"

    return metrics


def _calculate_recent_metrics(quantity_data, column_name, current_value):
    """计算近期指标"""
    metrics = {}

    quantity_values = quantity_data[column_name].dropna()
    if len(quantity_values) > 0:
        # 计算排名
        sorted_values = np.sort(quantity_values)
        rank = np.searchsorted(sorted_values, current_value, side='right')
        rank_percentage = (rank / len(quantity_values)) * 100

        metrics['近期排名'] = f"{rank}/{len(quantity_values)}"
        metrics['排名百分比'] = f"{rank_percentage:.2f}%"

        # 超量提示
        if rank_percentage >= 70:
            metrics['超量提示'] = "⚠️ 超量：处于近期前30%"
            metrics['超量等级'] = "高风险" if rank_percentage >= 90 else "中等风险"
        else:
            metrics['超量提示'] = "正常范围"
            metrics['超量等级'] = "低风险"

        # 统计信息
        metrics['近期统计'] = {
            '平均值': float(np.mean(quantity_values)),
            '中位数': float(np.median(quantity_values)),
            '最大值': float(np.max(quantity_values)),
            '最小值': float(np.min(quantity_values)),
            '标准差': float(np.std(quantity_values))
        }

    return metrics