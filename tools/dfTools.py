import pandas as pd
import numpy as np
import re

# 只组成明日涨跌幅
def reshape_stock_data(df, target_col='涨跌幅'):
    """
    将股票数据重组成包含明日涨跌幅的监督学习格式

    参数:
    df: 原始股票DataFrame，包含日期、价格、涨跌幅等列
    target_col: 要预测的目标列，默认为'涨跌幅'

    返回:
    reshaped_df: 重组成(a-1, 13)维度的新DataFrame
    """
    # 创建副本避免修改原数据
    df_work = df.copy()

    # 确保日期列正确排序
    if '日期' in df_work.columns:
        df_work = df_work.sort_values('日期').reset_index(drop=True)

    # 创建明日涨跌幅列（将涨跌幅列向前移动一位）
    df_work['明日涨跌幅'] = df_work[target_col].shift(-1)

    # 删除最后一行（因为明日涨跌幅为NaN）
    reshaped_df = df_work.iloc[:-1, :]

    print(f"原始数据形状: {df.shape}")
    print(f"重组后数据形状: {reshaped_df.shape}")

    return reshaped_df

# 灵活组成多日涨跌幅
def advanced_reshape_stock_data(df, feature_columns=None, target_col='涨跌幅', lookahead=1):
    """
    高级版本：支持多特征列和可调节预测步长

    参数:
    df: 原始股票DataFrame
    feature_columns: 要包含的特征列列表，如果为None则使用所有数值列
    target_col: 目标预测列
    lookahead: 预测步长（1表示明日，2表示后日等）
    """
    df_work = df.copy()

    # 确定特征列
    if feature_columns is None:
        # 自动选择数值列，排除目标列
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        feature_columns = numeric_cols

    # 确保日期排序
    if '日期' in df_work.columns:
        df_work = df_work.sort_values('日期').reset_index(drop=True)

    # 创建明日目标列
    new_target_name = f'{lookahead}日后{target_col}'
    df_work[new_target_name] = df_work[target_col].shift(-lookahead)

    # 删除最后lookahead行（包含NaN的行）
    reshaped_df = df_work.iloc[:-lookahead, :] if lookahead > 0 else df_work

    # 选择最终列（特征列 + 新目标列）
    final_columns = feature_columns + [new_target_name]
    if '日期' in df_work.columns:
        final_columns = ['日期'] + final_columns
    if '股票代码' in df_work.columns:
        final_columns = ['股票代码'] + final_columns

    final_df = reshaped_df[final_columns]

    print(f"特征列: {feature_columns}")
    print(f"目标列: {new_target_name}")
    print(f"最终数据形状: {final_df.shape}")

    return final_df