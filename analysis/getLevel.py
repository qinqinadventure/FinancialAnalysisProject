import analysis.LEVELanalysis.level as level


def outputLevelInfo(stock_data):
    # 假设 stock_data 是您的 DataFrame
    result = level.getLevel(stock_data)

    # 打印结果（可根据需要调整格式）
    print("=" * 50)
    print(f"分析日期: {result['分析时间']}")
    print("-" * 50)

    print("趋势分析:")
    trend = result['趋势分析']
    print(f"  收盘价: {trend['收盘价']:.2f}")
    print(f"  MA10: {trend['MA10']:.2f}")
    print(f"  MA20: {trend['MA20']:.2f}")
    print(f"  MA30: {trend['MA30']:.2f}")
    print(f"  趋势判断: {trend['趋势']}")

    print("\n压力位分析:")
    resistance = result['压力位分析']
    for key, value in resistance.items():
        if value is not None:
            print(f"  {key}: {value:.2f}")

    print("\n均线关系分析:")
    ma_rel = result['均线关系分析']
    for key, value in ma_rel.items():
        print(f"  {key}: {value}")

    print("=" * 50)