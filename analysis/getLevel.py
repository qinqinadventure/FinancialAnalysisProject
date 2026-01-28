import analysis.LEVELanalysis.level as level

def outputLevelInfo(stock_data, valuation_data=None, current_pe=None, current_pb=None):
    """
    输出完整的股票分析信息（优化版）
    """
    # 调用分析函数
    result = level.getLevel(stock_data, valuation_data, current_pe, current_pb)

    # 打印结果
    print("=" * 60)
    print(f"分析日期: {result['分析时间']}")
    print("-" * 60)

    # 1. 趋势分析
    print("📈 趋势分析:")
    trend = result['趋势分析']
    print(f"  收盘价: {trend['收盘价']:.2f}")
    print(f"  MA10: {trend['MA10']:.2f}")
    print(f"  MA20: {trend['MA20']:.2f}")
    print(f"  MA30: {trend['MA30']:.2f}")
    print(f"  趋势状态: {trend['趋势']}")

    # 2. 压力位分析
    print("\n🎯 压力位分析:")
    resistance = result['压力位分析']
    for key, value in resistance.items():
        if value is not None:
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: 无")

    # 3. 均线关系分析
    print("\n📊 均线关系分析:")
    ma_rel = result['均线关系分析']
    for key, value in ma_rel.items():
        print(f"  {key}: {value}")

    # 4. 历史位置分析（新增）
    print("\n📅 历史位置分析:")
    historical = result['历史位置分析']
    if isinstance(historical, dict):
        print(f"  当前价格: {historical.get('当前价格', 'N/A'):.2f}")
        print(f"  历史高点: {historical.get('历史高点', 'N/A'):.2f}")
        print(f"  历史低点: {historical.get('历史低点', 'N/A'):.2f}")
        print(f"  位置百分比: {historical.get('位置百分比', 'N/A')}%")
        print(f"  位置级别: {historical.get('位置级别', 'N/A')}")
        print(f"  分析周期: {historical.get('分析周期', 'N/A')}")
    else:
        print(f"  {historical}")

    # 5. 估值分析（新增）
    print("\n💰 估值分析:")
    valuation = result['估值分析']
    if valuation is not None:
        if '综合估值' in valuation:
            # 详细估值分析模式
            print(f"  PE分位: {valuation.get('PE分位', 'N/A')}")
            print(f"  PB分位: {valuation.get('PB分位', 'N/A')}")
            print(f"  综合估值: {valuation.get('综合估值', 'N/A')}")
            print(f"  当前PE: {valuation.get('当前PE', 'N/A'):.2f}")
            print(f"  当前PB: {valuation.get('当前PB', 'N/A'):.2f}")
        else:
            # 简化估值分析模式
            print(f"  PE水平: {valuation.get('PE水平', 'N/A')}")
            print(f"  PB水平: {valuation.get('PB水平', 'N/A')}")
            print(f"  当前PE: {valuation.get('当前PE', 'N/A'):.2f}")
            print(f"  当前PB: {valuation.get('当前PB', 'N/A'):.2f}")
    else:
        print("  无估值数据")

    # 6. 综合建议（新增）
    print("\n💡 综合建议:")
    print(f"  {result.get('综合建议', '暂无建议')}")

    print("=" * 60)

    return result