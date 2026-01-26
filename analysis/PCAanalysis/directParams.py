import pandas as pd
import numpy as np
import analysis.PCAanalysis.caculateParams as CP

def get_multiple_columns(df, column_names):
    """
    提取多个列的数据，支持混合普通列和计算表达式

    参数:
    df: pandas DataFrame
    column_names: 列名列表，可包含普通列名和计算表达式

    返回:
    pandas DataFrame: 包含指定列的新DataFrame
    """
    df_result = df.copy()
    valid_columns = []
    missing_columns = []

    for col in column_names:
        if CP.is_calculation_expression(col):
            # 处理计算表达式
            df_temp = CP.add_calculated_column(df_result, col)
            if df_temp is not None and col in df_temp.columns:
                df_result = df_temp
                valid_columns.append(col)
            else:
                print(f"计算表达式 '{col}' 处理失败")
        elif col in df_result.columns:
            # 普通列名存在
            valid_columns.append(col)
        else:
            # 列名不存在
            missing_columns.append(col)

    # 报告缺失的列
    if missing_columns:
        print(f"以下列名不存在: {missing_columns}")
        print(f"可用的列有: {list(df.columns)}")

    # 返回存在的列和成功计算的新列
    if valid_columns:
        return df_result[valid_columns]
    else:
        print("没有有效的列可提取")
        return None


def get_numeric_columns(df):
    """
    提取所有数值列（保持不变）
    """
    return df.select_dtypes(include=[np.number])


def get_date_related_columns(df):
    """
    提取与日期相关的列，支持计算表达式
    """
    date_related = ['日期', '开盘', '收盘', '最高', '最低', '涨跌幅', '涨跌额']
    return get_multiple_columns(df, date_related)


def get_volume_related_columns(df):
    """
    提取与成交量相关的列，支持计算表达式
    """
    volume_related = ['成交量', '成交额', '换手率']
    return get_multiple_columns(df, volume_related)


def smart_column_extractor(df, column_names):
    """
    智能列提取器：自动处理普通列和计算表达式

    参数:
    df: pandas DataFrame
    column_names: 列名列表，可混合普通列和计算表达式

    返回:
    tuple: (成功提取的DataFrame, 失败的列列表)
    """
    success_df = df.copy()
    failed_columns = []

    for col in column_names:
        if CP.is_calculation_expression(col):
            # 尝试添加计算列
            temp_df = CP.add_calculated_column(success_df, col)
            if temp_df is not None and col in temp_df.columns:
                success_df = temp_df
            else:
                failed_columns.append(col)
                print(f"计算列 '{col}' 添加失败")
        elif col not in success_df.columns:
            failed_columns.append(col)
            print(f"列 '{col}' 不存在")

    # 提取成功的列
    success_columns = [col for col in column_names if col not in failed_columns]

    if success_columns:
        result_df = success_df[success_columns]
        return result_df, failed_columns
    else:
        print("没有列成功提取")
        return None, failed_columns