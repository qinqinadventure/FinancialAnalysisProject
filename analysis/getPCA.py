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

    # 计算单个特征的相似度
    comp_score, detailed_scores = similarity_analyzer.comprehensive_similarity(0)
    print("详细得分:", {k: f"{v:.4f}" for k, v in detailed_scores.items()})

    # 对所有特征进行排序
    rankings = similarity_analyzer.rank_features()
    print(f"\n特征相似度排名:")
    for i, (feature_idx, score) in enumerate(rankings):
        print(f"{i + 1}. {feature_idx}: {score:.4f}")