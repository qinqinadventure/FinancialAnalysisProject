import analysis.PCAanalysis.directParams as DP
import tools.dfTools as DT
import analysis.PCA as PCA

# 根据名称一个个获取对应参数
def getFeatures(features,stockdata):
    return DP.get_multiple_columns(stockdata,features)

# 根据传入的数据获取PCA分析结果
def PCAResult(featuresName,stockdata):
    # 将涨跌幅添加到最后一列
    reDF = DT.reshape_stock_data(stockdata)
    # 添加最后一列
    addFeatures = featuresName + ["明日涨跌幅"]
    # 首先添加要分析的列
    addDF = getFeatures(addFeatures,reDF)
    # 进行分析
    # 生成数据
    features = DT.extract_columns_to_ndarray(addDF,featuresName)
    target = DT.extract_columns_to_ndarray(addDF,"明日涨跌幅")
    # 创建相似度分析器
    similarity_analyzer = PCA.PCASimilarity(n_components=3, feature_names=featuresName)

    # 拟合模型
    similarity_analyzer.fit(features, target)

    # 获取PCA摘要
    summary = similarity_analyzer.get_pca_summary()

    print("PCA模型摘要:")
    print(f"主成分数量: {summary['n_components']}")
    print(f"方差解释比例: {summary['explained_variance_ratio']}")
    print(f"累计方差解释: {summary['cumulative_variance_ratio']}")

    # 对所有特征进行排序
    rankings = similarity_analyzer.rank_features()
    print(f"\n特征相似度排名:")
    for i, (feature_name, score) in enumerate(rankings):
        print(f"{i + 1}. {feature_name}: {score:.4f}")

    print("=" * 50)
    print("📈 PCA分析详细报告")
    print("=" * 50)

    print("🔍 模型基本信息:")
    print(f"   主成分数量: {summary['n_components']}")
    print(f"   原始特征数: {len(featuresName)}")
    print(f"   降维比例: {summary['n_components']}/{len(featuresName)}")

    print("\n📊 方差解释分析:")
    # 使用特征名称而不是数字编号
    for i, (variance, cumulative) in enumerate(zip(
            summary['explained_variance_ratio'],
            summary['cumulative_variance_ratio']
    )):
        print(f"   维度 {i + 1}: {variance:.1%} (累计: {cumulative:.1%})")

    # 新增：显示每个主成分的主要贡献特征
    print("\n🔬 主成分特征贡献分析:")
    components = similarity_analyzer.components_
    feature_names = featuresName + ["明日涨跌幅"]  # 包含目标变量

    for i in range(summary['n_components']):
        print(f"\n   主成分 {i + 1} (解释方差: {summary['explained_variance_ratio'][i]:.1%}):")

        # 获取当前主成分的载荷向量
        loadings = components[i]

        # 创建特征-载荷对的列表并排序
        feature_loadings = []
        for j, feature_name in enumerate(feature_names):
            feature_loadings.append((feature_name, loadings[j]))

        # 按载荷绝对值降序排序
        feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)

        # 显示前3个最重要的特征
        for k in range(min(3, len(feature_loadings))):
            feature, loading = feature_loadings[k]
            direction = "正向" if loading > 0 else "负向"
            print(f"     {k + 1}. {feature}: {loading:.3f} ({direction}贡献)")

    print("\n💡 结果解读:")
    total_variance = summary['cumulative_variance_ratio'][-1]
    if total_variance > 0.8:
        print(f"   ✅ 优秀: 前{summary['n_components']}个主成分保留了{total_variance:.1%}的信息")
        print("   建议: 当前主成分数量设置合理")
    elif total_variance > 0.7:
        print(f"   ⚠️ 良好: 保留了{total_variance:.1%}的信息，可考虑增加主成分")
    else:
        print(f"   ❌ 不足: 仅保留{total_variance:.1%}的信息")
        print("   建议: 增加主成分数量或检查数据质量")

    print("\n🎯 投资洞察:")
    # 基于主成分分析提供投资建议
    if len(rankings) > 0:
        top_feature = rankings[0][0]  # 排名第一的特征
        print(f"   最相关特征: '{top_feature}' 与明日涨跌幅关联度最高")
        print("   建议重点关注该指标的走势变化")

    print("=" * 50)

    return similarity_analyzer