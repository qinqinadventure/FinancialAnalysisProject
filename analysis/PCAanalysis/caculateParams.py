import numpy as np
import re

# 支持根据传入的字符进行四则运算
def add_calculated_column(df, expression, new_column_name=None):
    """
    为DataFrame添加通过四则运算计算的新列

    参数:
    df: pandas DataFrame
    expression: 运算表达式，例如 "涨幅/成交量"、"收盘价*成交量"
    new_column_name: 新列名，如果为None则使用表达式作为列名

    返回:
    DataFrame: 添加了新列的DataFrame副本
    """
    # 创建副本避免修改原数据
    df_result = df.copy()

    # 如果没有指定新列名，使用表达式
    if new_column_name is None:
        new_column_name = expression

    try:
        # 提取表达式中所有可能的列名（连续的中文、英文、数字和下划线）
        potential_columns = re.findall(r'[\w\u4e00-\u9fff]+', expression)

        # 过滤出实际存在于DataFrame中的列
        valid_columns = [col for col in potential_columns if col in df.columns]

        if len(valid_columns) < 2:
            raise ValueError(f"表达式中需要至少两个有效列名，找到的列: {valid_columns}")

        # 安全地执行运算
        # 将表达式中的列名用df['列名']替换
        safe_expression = expression
        for col in valid_columns:
            safe_expression = safe_expression.replace(col, f"df_result['{col}']")

        # 执行计算
        new_column = eval(safe_expression)

        # 添加到DataFrame
        df_result[new_column_name] = new_column

        print(f"成功添加列: {new_column_name} = {expression}")

    except Exception as e:
        print(f"计算失败: {e}")
        print(f"请检查表达式 '{expression}' 中的列名是否正确")
        print(f"可用的列有: {list(df.columns)}")

    return df_result