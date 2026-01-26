import analysis.PCA as PCA
import numpy as np

# 示例数据生成
def generate_sample_data(n_samples=1000, n_features=10):
    """生成示例数据"""
    np.random.seed(42)

    # 生成基础特征
    data = np.random.randn(n_samples, n_features)

    # 创建目标变量（与某些特征相关）
    target = (data[:, 0] * 0.7 + data[:, 2] * 0.3 +
              data[:, 5] * 0.5 + np.random.randn(n_samples) * 0.1)

    return data, target


# 使用示例
if __name__ == "__main__":
    # 生成数据
    features, target = generate_sample_data()

    # 示例：使用特征名称
    feature_names = ['开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额',
                     '换手率', '市盈率', '市净率', '流通市值']

    # 创建相似度分析器
    similarity_analyzer = PCA.PCASimilarity(n_components=3, feature_names=feature_names)

    # 拟合模型
    # 一千个，十个维度
    similarity_analyzer.fit(features, target)

    # 获取PCA摘要
    summary = similarity_analyzer.get_pca_summary()
    print("PCA模型摘要:")
    print(f"主成分数量: {summary['n_components']}")
    print(f"方差解释比例: {summary['explained_variance_ratio']}")
    print(f"累计方差解释: {summary['cumulative_variance_ratio']}")

    # 计算单个特征的相似度
    comp_score, detailed_scores = similarity_analyzer.comprehensive_similarity(0)
    print(f"\n特征0的综合相似度: {comp_score:.4f}")
    print("详细得分:", {k: f"{v:.4f}" for k, v in detailed_scores.items()})

    # 对所有特征进行排序
    rankings = similarity_analyzer.rank_features()
    print(f"\n特征相似度排名:")
    for i, (feature_idx, score) in enumerate(rankings):
        print(f"{i + 1}. 特征{feature_idx}: {score:.4f}")