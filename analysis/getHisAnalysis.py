import analysis.HISAnalysis.getHis as getHis


def outputHisAnalysis(history_data, quantity_data, analysis_target):
    """
    优化版的历史分位分析结果输出函数
    提供清晰、结构化的分析报告
    """
    output = getHis.getHisAnalysis(history_data, quantity_data, analysis_target)

    # 打印报告头部
    print("🔍" + "=" * 70 + "🔍")
    print(" " * 25 + "历史分位分析报告")
    print("🔍" + "=" * 70 + "🔍")

    # 总体信息
    if '总体信息' in output:
        overall = output['总体信息']
        print(f"\n📊 分析概况:")
        print(f"   • 分析目标: {', '.join(overall.get('分析目标列表', []))}")
        print(f"   • 成功分析: {overall.get('成功分析数量', 0)}/{overall.get('分析目标数量', 0)}")
        print(f"   • 历史数据: {overall.get('历史数据形状', '未知')}条记录")
        print(f"   • 近期数据: {overall.get('近期数据形状', '未知')}个交易日")

    # 逐个分析目标展示
    for target_name, result in output.items():
        if target_name == '总体信息':
            continue

        print(f"\n🎯" + "-" * 60 + "🎯")
        print(f"📈 分析指标: {result.get('分析目标', target_name)}")
        print("🎯" + "-" * 60 + "🎯")

        if '错误信息' in result:
            print(f"❌ 分析失败: {result['错误信息']}")
            continue

        # 当前值和基本统计
        current_val = result.get('当前值', 0)
        print(f"💰 当前值: {current_val:,.4f}")

        # 历史位置分析（重点突出）
        print(f"\n📅 历史位置分析（基于{result.get('数据信息', {}).get('历史数据量', 0)}条历史数据）:")
        percentile = result.get('历史分位', 'N/A')
        level = result.get('历史水平', 'N/A')

        # 使用颜色标识历史水平
        level_icon = "🚨" if "极高" in level else "⚠️" if "高" in level else "✅" if "低" in level else "📊"
        print(f"   {level_icon} 历史分位: {percentile} - {level}")
        print(f"   • 高于历史比例: {result.get('高于历史比例', 'N/A')}")
        print(f"   • 低于历史比例: {result.get('低于历史比例', 'N/A')}")

        # 近期表现分析（明确说明时间范围）
        recent_days = result.get('数据信息', {}).get('近期数据量', 15)
        print(f"\n🔄 近期表现分析（最近{recent_days}个交易日）:")
        rank_info = result.get('近期排名', 'N/A')
        rank_percent = result.get('排名百分比', 'N/A')
        alert = result.get('超量提示', 'N/A')
        alert_level = result.get('超量等级', 'N/A')

        print(f"   📊 近期排名: {rank_info}（当前值在最近{recent_days}天中排名第{rank_info.split('/')[0]}位）")
        print(f"   📈 排名百分比: {rank_percent}")

        # 风险提示（明确说明含义）
        alert_icon = "🚨" if "高风险" in alert_level else "⚠️" if "中等风险" in alert_level else "✅"
        print(f"   {alert_icon} 风险提示: {alert}")
        print(f"   • 风险等级: {alert_level}")

        # 近期统计信息
        if '近期统计' in result:
            stats = result['近期统计']
            print(f"\n📋 近期统计信息（最近{recent_days}个交易日）:")
            # 主要修改：移除了数字格式中的逗号（,），避免千分位分隔符在小数场景下的问题
            print(f"   • 平均值: {stats.get('平均值', 'N/A'):.4f}")
            print(f"   • 中位数: {stats.get('中位数', 'N/A'):.4f}")
            print(f"   • 最大值: {stats.get('最大值', 'N/A'):.4f}")
            print(f"   • 最小值: {stats.get('最小值', 'N/A'):.4f}")
            print(f"   • 标准差: {stats.get('标准差', 'N/A'):.4f}")

            # 当前值与平均值比较
            if isinstance(stats.get('平均值'), (int, float)) and current_val != 0:
                diff_ratio = (current_val - stats['平均值']) / abs(stats['平均值']) * 100
                diff_icon = "📈" if diff_ratio > 0 else "📉"
                # 主要修改：使用更精确的格式来显示微小的百分比变化
                # 方案1：增加小数位数（例如6位）
                print(f"   {diff_icon} 当前值较近期平均: {diff_ratio:+.6f}%")

                # 或者 方案2：对于极小的值，使用科学计数法显示会更清晰
                # if abs(diff_ratio) < 0.01: # 如果差值非常小
                #     print(f"   {diff_icon} 当前值较近期平均: {diff_ratio:+.2e}%")
                # else:
                #     print(f"   {diff_icon} 当前值较近期平均: {diff_ratio:+.2f}%")

                # 调试语句：如果需要，可以打印原始值以确认（正式版可删除）
                # print(f"   [调试] 当前值: {current_val}, 平均值: {stats['平均值']}, 原始差值比率: {diff_ratio}")

        # 数据质量信息
        if '数据信息' in result:
            data_info = result['数据信息']
            print(f"\n💾 数据质量:")
            print(f"   • 历史数据量: {data_info.get('历史数据量', 'N/A')}条")
            print(f"   • 近期数据量: {data_info.get('近期数据量', 'N/A')}天")
            if '使用列名' in data_info:
                print(f"   • 计算使用列: {', '.join(data_info['使用列名'])}")

    # 综合评估和建议
    print("\n💡" + "=" * 60 + "💡")
    print(" " * 22 + "综合评估与投资建议")
    print("💡" + "=" * 60 + "💡")

    successful_targets = [(name, output[name]) for name in output.keys()
                          if name != '总体信息' and '错误信息' not in output[name]]

    if successful_targets:
        print("✅ 成功分析指标:")
        for target_name, result in successful_targets:
            percentile = float(result.get('历史分位', '0%').rstrip('%'))
            rank_percent = float(result.get('排名百分比', '0%').rstrip('%'))

            # 生成针对性的评估
            level_icon = "🚨" if percentile >= 80 else "💡" if percentile <= 20 else "📊"
            trend_icon = "🔥" if rank_percent >= 80 else "📈" if rank_percent >= 60 else "📉"

            print(f"   {level_icon} {target_name}:")
            print(f"     {trend_icon} 历史位置: {result.get('历史水平', 'N/A')}")
            print(f"     📊 近期强度: 排名前{100 - rank_percent:.0f}%")

            # 生成建议
            if percentile >= 80 and rank_percent >= 70:
                print(f"     ⚠️  建议: 历史高位+近期强势，注意回调风险")
            elif percentile <= 20 and rank_percent >= 70:
                print(f"     💡 建议: 历史低位+近期走强，可能存在机会")
            elif percentile >= 80:
                print(f"     ⚠️  建议: 处于历史高位区域，谨慎操作")
            elif percentile <= 20:
                print(f"     💡 建议: 处于历史低位区域，值得关注")
            else:
                print(f"     🔄 建议: 历史位置合理，结合其他指标判断")
            print()
    else:
        print("❌ 没有成功分析的指标")

    print("🔚" + "=" * 70 + "🔚")
    print(" " * 28 + "分析结束")
    print("🔚" + "=" * 70 + "🔚")

    return output