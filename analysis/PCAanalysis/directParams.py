import pandas as pd
import numpy as np


def get_column_by_name(df, column_name):
    """
    根据列名提取单列数据

    参数:
    df: pandas DataFrame
    column_name: 列名字符串

    返回:
    pandas Series: 指定列的数据
    """
    if column_name in df.columns:
        return df[column_name]
    else:
        print(f"列名 '{column_name}' 不存在。可用的列有: {list(df.columns)}")
        return None


def get_multiple_columns(df, column_names):
    """
    提取多个列的数据

    参数:
    df: pandas DataFrame
    column_names: 列名列表

    返回:
    pandas DataFrame: 包含指定列的新DataFrame
    """
    missing_cols = [col for col in column_names if col not in df.columns]
    if missing_cols:
        print(f"以下列名不存在: {missing_cols}")
        print(f"可用的列有: {list(df.columns)}")
        return None

    return df[column_names]


def get_numeric_columns(df):
    """
    提取所有数值列
    """
    return df.select_dtypes(include=[np.number])


def get_date_related_columns(df):
    """
    提取与日期相关的列
    """
    date_related = ['日期', '开盘', '收盘', '最高', '最低', '涨跌幅', '涨跌额']
    return get_multiple_columns(df, [col for col in date_related if col in df.columns])


def get_volume_related_columns(df):
    """
    提取与成交量相关的列
    """
    volume_related = ['成交量', '成交额', '换手率']
    return get_multiple_columns(df, [col for col in volume_related if col in df.columns])