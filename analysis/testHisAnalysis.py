import numpy as np
import pandas as pd
import getHisAnalysis

# 使用示例
def example_usage():
    """
    使用示例 - 演示如何处理多个分析目标
    """
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1329, freq='D')

    # 创建history_data
    history_data = pd.DataFrame({
        '日期': dates,
        '收盘价': np.random.randn(1329) * 10 + 100,
        '成交量': np.random.randint(10000, 1000000, 1329),
        '涨跌幅': np.random.randn(1329) * 0.05
    })

    # 创建quantity_data
    recent_dates = pd.date_range('2024-12-01', periods=15, freq='D')
    quantity_data = pd.DataFrame({
        '日期': recent_dates,
        '收盘价': np.random.randn(15) * 5 + 105,
        '成交量': np.random.randint(50000, 2000000, 15),
        '涨跌幅': np.random.randn(15) * 0.03
    })

    # 测试单个目标
    print("=== 单个分析目标 ===")
    result_single = getHisAnalysis.outputHisAnalysis(history_data, quantity_data, '收盘价')
    print(f"分析结果键: {list(result_single.keys())}")

    # 测试多个目标
    print("\n=== 多个分析目标 ===")
    result_multi = getHisAnalysis.outputHisAnalysis(history_data, quantity_data, ['收盘价', '成交量', '涨跌幅'])

    # 打印总体信息
    overall = result_multi['总体信息']
    print(f"分析目标数量: {overall['分析目标数量']}")
    print(f"成功分析数量: {overall['成功分析数量']}")
    print(f"分析目标列表: {overall['分析目标列表']}")

    # 打印每个目标的简要结果
    for target in overall['分析目标列表']:
        if target in result_multi and '错误信息' not in result_multi[target]:
            target_result = result_multi[target]
            print(f"\n{target}:")
            print(f"  当前值: {target_result.get('当前值', 'N/A')}")
            print(f"  历史分位: {target_result.get('历史分位', 'N/A')}")
            print(f"  历史水平: {target_result.get('历史水平', 'N/A')}")
        else:
            print(f"\n{target}: 分析失败")

    return result_multi


if __name__ == "__main__":
    example_result = example_usage()