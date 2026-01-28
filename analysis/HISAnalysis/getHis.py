import numpy as np
import analysis.PCAanalysis.directParams as DP
import pandas as pd


def getHisAnalysis(history_data, quantity_data, analysis_target):
    """
    历史分位分析函数 - 支持analysis_target为列表
    参数:
        history_data: 历史数据DataFrame，形状为(1329, 12)
        quantity_data: 近期数据DataFrame，形状为(15, 12)
        analysis_target: 分析目标列名，可以是字符串或字符串列表

    返回:
        dict: 包含分位分析和排名信息的字典
    """
    results = {}

    # 确保analysis_target是列表格式[1,2](@ref)
    if isinstance(analysis_target, str):
        analysis_target = [analysis_target]

    # 对每个分析目标分别处理
    for target in analysis_target:
        target_results = {}

        try:
            # 调用DT.extract_columns_to_ndarray获取数据[1](@ref)
            add_history_data = DP.get_multiple_columns(history_data, target)
            add_quantity_data = DP.get_multiple_columns(quantity_data, target)

            # 将新列添加到原来的DF[7](@ref)
            # 注意：这里应该是列合并，不是行合并
            history_data_with_target = pd.concat([history_data, add_history_data], axis=1)
            quantity_data_with_target = pd.concat([quantity_data, add_quantity_data], axis=1)

            # 获取当前值（quantity_data中的最新值）[2](@ref)
            if len(quantity_data_with_target) > 0:
                if hasattr(quantity_data_with_target, 'iloc'):
                    current_value = quantity_data_with_target[target].iloc[-1]
                else:
                    current_value = quantity_data_with_target[target][-1]
            else:
                current_value = quantity_data[target].iloc[-1]

            # 1. 获取当前位置的历史分位
            if '日期' in history_data_with_target.columns:
                history_data_sorted = history_data_with_target.sort_values('日期')
            else:
                history_data_sorted = history_data_with_target

            history_values = history_data_sorted[target].dropna()

            if len(history_values) > 0:
                # 计算当前值在历史数据中的分位[3](@ref)
                percentile_rank = np.sum(history_values <= current_value) / len(history_values) * 100

                target_results['分析目标'] = target
                target_results['当前值'] = float(current_value)
                target_results['历史分位'] = f"{percentile_rank:.2f}%"
                target_results['高于历史比例'] = f"{percentile_rank:.2f}%"
                target_results['低于历史比例'] = f"{(100 - percentile_rank):.2f}%"

                # 判断历史水平
                if percentile_rank >= 90:
                    target_results['历史水平'] = "极高水平(前10%)"
                elif percentile_rank >= 70:
                    target_results['历史水平'] = "高水平(前30%)"
                elif percentile_rank >= 30:
                    target_results['历史水平'] = "中等水平"
                elif percentile_rank >= 10:
                    target_results['历史水平'] = "低水平(后30%)"
                else:
                    target_results['历史水平'] = "极低水平(后10%)"

            # 2. 获取指定指标在近quantity_data的rank
            if len(quantity_data_with_target) > 0:
                # 获取quantity_data期间的所有值
                quantity_values = quantity_data_with_target[target].values

                # 过滤有效值
                valid_values = quantity_values[~np.isnan(quantity_values)]

                if len(valid_values) > 0:
                    # 计算当前值的排名[4](@ref)
                    sorted_values = np.sort(valid_values)
                    rank = np.searchsorted(sorted_values, current_value, side='right')
                    rank_percentage = (rank / len(valid_values)) * 100

                    target_results['近期排名'] = f"{rank}/{len(valid_values)}"
                    target_results['排名百分比'] = f"{rank_percentage:.2f}%"

                    # 判断是否超量（前30%）[4](@ref)
                    if rank_percentage >= 70:
                        target_results['超量提示'] = "⚠️ 超量：处于近期前30%"
                        target_results['超量等级'] = "高风险" if rank_percentage >= 90 else "中等风险"
                    else:
                        target_results['超量提示'] = "正常范围"
                        target_results['超量等级'] = "低风险"

                    # 添加统计信息
                    target_results['近期统计'] = {
                        '平均值': float(np.mean(valid_values)),
                        '中位数': float(np.median(valid_values)),
                        '最大值': float(np.max(valid_values)),
                        '最小值': float(np.min(valid_values)),
                        '标准差': float(np.std(valid_values))
                    }

            # 3. 添加数据信息
            target_results['数据信息'] = {
                '历史数据量': len(history_values),
                '近期数据量': len(valid_values) if 'valid_values' in locals() else 0,
                '目标列存在': target in history_data.columns and target in quantity_data.columns
            }

        except Exception as e:
            target_results = {
                '分析目标': target,
                '错误信息': f"处理失败: {str(e)}",
                '数据信息': {
                    '目标列存在': target in history_data.columns and target in quantity_data.columns
                }
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