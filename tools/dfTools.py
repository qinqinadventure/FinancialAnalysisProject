import pandas as pd
import numpy as np
import re
from typing import Union, List

# 只组成明日涨跌幅
def reshape_stock_data(df, target_col='涨跌幅'):
    """
    重组成包含明日涨跌幅的监督学习格式（修复版本）

    参数:
    df: 原始股票DataFrame
    target_col: 目标列名，默认为'涨跌幅'

    返回:
    DataFrame: 重组成(a-1, n+1)维度的新DataFrame
    """
    try:
        # 1. 验证输入数据
        if df is None or df.empty:
            print("错误: 输入数据为空")
            return None

        # 创建副本避免修改原数据
        df_work = df.copy()

        # 2. 验证target_col参数
        if target_col is None or target_col == '':
            print("错误: target_col 参数不能为空")
            return df_work

        # 3. 检查目标列是否存在
        if target_col not in df_work.columns:
            print(f"错误: 列 '{target_col}' 不存在于DataFrame中")
            print(f"可用的列有: {list(df_work.columns)}")
            return df_work

        # 4. 检查目标列数据类型并转换
        if not pd.api.types.is_numeric_dtype(df_work[target_col]):
            print(f"警告: 列 '{target_col}' 不是数值类型，尝试转换...")
            try:
                df_work[target_col] = pd.to_numeric(df_work[target_col], errors='coerce')
                # 检查转换后是否还有有效数据
                if df_work[target_col].isna().all():
                    print(f"错误: 列 '{target_col}' 转换后全部为NaN")
                    return df_work
            except Exception as e:
                print(f"数据类型转换失败: {e}")
                return df_work

        # 5. 确保数据按日期排序（如果有日期列）
        if '日期' in df_work.columns:
            try:
                df_work = df_work.sort_values('日期').reset_index(drop=True)
                print("数据已按日期排序")
            except Exception as e:
                print(f"日期排序失败: {e}")

        # 6. 创建明日涨跌幅列
        try:
            # 方法1: 直接使用shift
            df_work['明日涨跌幅'] = df_work[target_col].shift(-1)
            print(f"成功创建'明日涨跌幅'列，基于列: {target_col}")

        except Exception as shift_error:
            print(f"shift操作失败: {shift_error}")
            # 方法2: 使用手动循环（更安全但较慢）
            print("尝试使用备用方法...")
            tomorrow_values = []
            for i in range(len(df_work)):
                if i < len(df_work) - 1:
                    tomorrow_values.append(df_work[target_col].iloc[i + 1])
                else:
                    tomorrow_values.append(np.nan)

            df_work['明日涨跌幅'] = tomorrow_values
            print("备用方法成功")

        # 7. 删除包含NaN的行（最后一行）
        original_len = len(df_work)

        # 修复：使用明确的布尔判断条件
        nan_mask = df_work['明日涨跌幅'].isna()
        rows_to_keep = ~nan_mask  # 取反，保留非NaN的行

        # 使用布尔索引而不是直接判断DataFrame
        df_work = df_work[rows_to_keep].reset_index(drop=True)

        removed_count = original_len - len(df_work)
        print(f"删除了 {removed_count} 行包含NaN的数据")
        print(f"重组后数据形状: {df_work.shape}")

        # 8. 使用正确的方法检查DataFrame是否为空
        if df_work.empty:  # 使用.empty属性而不是直接判断
            print("警告: 重组后数据为空")
            return df_work

        # 9. 验证结果
        print("\n重组后数据预览:")
        preview_cols = [target_col, '明日涨跌幅']
        if '日期' in df_work.columns:
            preview_cols = ['日期'] + preview_cols
        print(df_work[preview_cols].head())

        return df_work

    except Exception as e:
        print(f"reshape_stock_data函数执行失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        # 返回原始数据而不是None，避免后续处理出错
        return df if df is not None else pd.DataFrame()


def safe_dataframe_check(df):
    """
    安全的DataFrame检查函数
    """
    if df is None:
        return False
    elif hasattr(df, 'empty'):
        return not df.empty
    else:
        return len(df) > 0

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


def extract_columns_to_ndarray(df: pd.DataFrame,
                               features: Union[str, List[str]]) -> np.ndarray:
    """
    从DataFrame中根据列名抽取数据构成ndarray

    参数:
    df: pandas DataFrame
    features: 列名字符串或列名列表

    返回:
    numpy ndarray: 包含指定列数据的数组
    """
    try:
        # 处理单列情况
        if isinstance(features, str):
            if features not in df.columns:
                raise ValueError(f"列名 '{features}' 不存在。可用的列有: {list(df.columns)}")

            # 提取单列并转换为ndarray (保持二维形状)
            result = df[[features]].values
            print(f"成功提取单列 '{features}'，形状: {result.shape}")
            return result

        # 处理多列情况
        elif isinstance(features, list):
            if not features:
                raise ValueError("特征列表不能为空")

            # 检查所有列是否存在
            missing_cols = [col for col in features if col not in df.columns]
            if missing_cols:
                raise ValueError(f"以下列名不存在: {missing_cols}。可用的列有: {list(df.columns)}")

            # 提取多列并转换为ndarray
            result = df[features].values
            print(f"成功提取 {len(features)} 列，形状: {result.shape}")
            return result

        else:
            raise TypeError("features参数必须是字符串或列表")

    except Exception as e:
        print(f"数据提取失败: {e}")
        return np.array([])  # 返回空数组
