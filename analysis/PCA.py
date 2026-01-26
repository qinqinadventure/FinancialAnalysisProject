import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr


class PCASimilarity:
    """
    基于PCA的多数据列与目标数据列相似度分析类（改进版本）
    """

    def __init__(self, n_components=2, standardize=True, feature_names=None):
        self.n_components = n_components
        self.standardize = standardize
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() if standardize else None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.feature_names = feature_names

    def fit(self, data_columns, target_column, target_name="target"):
        """
        拟合PCA模型到数据
        """
        if isinstance(data_columns, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = data_columns.columns.tolist()
            data_columns = data_columns.values

        if isinstance(target_column, pd.Series):
            target_column = target_column.values

        # 存储目标名称
        self.target_name = target_name

        # 将目标列作为额外特征合并
        target_reshaped = target_column.reshape(-1, 1)
        self.combined_data = np.hstack([data_columns, target_reshaped])

        # 如果没有提供特征名称，创建默认名称
        if self.feature_names is None:
            n_features = data_columns.shape[1]
            self.feature_names = [f'特征{i}' for i in range(n_features)]

        # 添加目标名称到特征名称列表
        self.all_feature_names = self.feature_names + [self.target_name]

        # 数据标准化
        if self.standardize:
            self.combined_data_scaled = self.scaler.fit_transform(self.combined_data)
        else:
            self.combined_data_scaled = self.combined_data

        # PCA拟合
        self.pca.fit(self.combined_data_scaled)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_

        # 获取降维后的数据
        self.transformed_data = self.pca.transform(self.combined_data_scaled)

        return self

    def get_feature_name(self, feature_idx):
        """根据索引获取特征名称"""
        if feature_idx < len(self.all_feature_names):
            return self.all_feature_names[feature_idx]
        return f"特征{feature_idx}"

    def get_correlation_direction(self, feature_idx, target_idx=-1):
        """
        获取特征与目标的相关性方向（正相关/负相关）
        返回: 1表示正相关，-1表示负相关，0表示无相关或无法判断
        """
        try:
            feature_data = self.combined_data[:, feature_idx]
            target_data = self.combined_data[:, target_idx]

            corr, p_value = pearsonr(feature_data, target_data)

            if abs(corr) < 0.1:  # 相关性很弱，认为无方向
                return 0
            elif corr > 0:
                return 1  # 正相关
            else:
                return -1  # 负相关
        except Exception as e:
            print(f"获取相关性方向失败: {e}")
            return 0

    def rank_features(self, target_idx=-1, method='comprehensive'):
        """
        对所有特征列进行相似度排序（修复版本）
        """
        try:
            n_features = self.combined_data.shape[1] - 1  # 排除目标列
            rankings = []

            for i in range(n_features):
                if method == 'comprehensive':
                    # 方法1：接收所有返回值
                    result = self.comprehensive_similarity(i, target_idx)

                    # 根据实际返回值数量处理
                    if len(result) == 3:
                        score, detailed_scores, direction = result
                    elif len(result) == 2:
                        score, detailed_scores = result
                    else:
                        score = result[0] if isinstance(result, (tuple, list)) else result

                elif method == 'pca_cosine':
                    score = self.cosine_similarity_pca(i, target_idx)
                elif method == 'variance_weighted':
                    score = self.explained_variance_similarity(i, target_idx)
                elif method == 'correlation':
                    score = self.correlation_similarity(i, target_idx)
                else:
                    score = self.distance_similarity(i, target_idx)

                # 使用特征名称而不是数字索引
                feature_name = self.get_feature_name(i)
                rankings.append((feature_name, score))

            # 按相似度降序排序
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings

        except Exception as e:
            print(f"rank_features计算错误: {e}")
            return []

    def explained_variance_similarity(self, feature_idx, target_idx=-1):
        """
        基于方差解释比重的加权相似度（改进版本）
        """
        try:
            if self.components_ is None or self.explained_variance_ratio_ is None:
                return 0

            weighted_similarity = 0
            total_variance = 0

            for i in range(min(self.n_components, len(self.explained_variance_ratio_))):
                try:
                    # 使用PCA载荷计算相似度（考虑方向）
                    feature_loading = self.components_[i, feature_idx]
                    target_loading = self.components_[i, target_idx]

                    # 计算方向感知的相似度（保留符号信息）
                    loading_similarity = feature_loading * target_loading

                    weight = self.explained_variance_ratio_[i]
                    weighted_similarity += abs(loading_similarity) * weight
                    total_variance += weight

                except Exception as dim_error:
                    continue

            similarity = weighted_similarity / total_variance if total_variance > 0 else 0

            # 根据相关性方向调整符号
            direction = self.get_correlation_direction(feature_idx, target_idx)
            return similarity * direction if direction != 0 else similarity

        except Exception as e:
            print(f"explained_variance_similarity计算错误: {e}")
            return 0

    def cosine_similarity_pca(self, feature_idx, target_idx=-1):
        """
        在PCA空间计算特征与目标之间的余弦相似度（改进版本）
        """
        try:
            feature_pca_loading = self.components_[:, feature_idx]
            target_pca_loading = self.components_[:, target_idx]

            # 计算余弦相似度（保留方向信息）
            cosine_sim = 1 - cosine(feature_pca_loading, target_pca_loading)

            # 根据原始数据相关性调整方向
            direction = self.get_correlation_direction(feature_idx, target_idx)
            return cosine_sim * direction if direction != 0 else max(0, cosine_sim)

        except Exception as e:
            print(f"cosine_similarity_pca计算错误: {e}")
            return 0

    def correlation_similarity_with_direction(self, feature_idx, target_idx=-1):
        """
        基于相关系数的相似度计算（保留方向信息）
        返回: (相似度得分, 方向)
        """
        try:
            feature_data = self.combined_data[:, feature_idx]
            target_data = self.combined_data[:, target_idx]

            corr, p_value = pearsonr(feature_data, target_data)
            return abs(corr), 1 if corr > 0 else -1

        except Exception as e:
            print(f"correlation_similarity计算错误: {e}")
            return 0, 0

    def comprehensive_similarity(self, feature_idx, target_idx=-1, weights=None):
        """
        综合相似度评估（改进版本，包含方向信息）
        """
        if weights is None:
            weights = {
                'pca_cosine': 0.3,
                'variance_weighted': 0.3,
                'distance': 0.2,
                'correlation': 0.2
            }

        try:
            # 获取方向信息
            direction = self.get_correlation_direction(feature_idx, target_idx)

            # 计算各指标得分
            scores = {}
            scores['pca_cosine'] = self.cosine_similarity_pca(feature_idx, target_idx)
            scores['variance_weighted'] = self.explained_variance_similarity(feature_idx, target_idx)
            scores['distance'] = self.distance_similarity(feature_idx, target_idx)

            # 相关系数单独处理（保留方向）
            corr_score, corr_direction = self.correlation_similarity_with_direction(feature_idx, target_idx)
            scores['correlation'] = corr_score

            # 计算加权综合得分
            composite_score = 0
            for key, weight in weights.items():
                composite_score += abs(scores[key]) * weight

            # 应用主要方向（以相关系数方向为主）
            if direction != 0:
                composite_score *= direction
            elif corr_direction != 0:
                composite_score *= corr_direction

            return composite_score, scores, direction

        except Exception as e:
            print(f"comprehensive_similarity计算错误: {e}")
            return 0, {}, 0

    def distance_similarity(self, feature_idx, target_idx=-1, metric='euclidean'):
        """
        基于距离的相似度计算（保持不变）
        """
        try:
            feature_data = self.combined_data_scaled[:, feature_idx]
            target_data = self.combined_data_scaled[:, target_idx]

            if metric == 'euclidean':
                distance = euclidean(feature_data, target_data)
                similarity = 1 / (1 + distance)
            elif metric == 'cosine':
                distance = cosine(feature_data, target_data)
                similarity = 1 - distance
            else:
                similarity = 0

            return similarity
        except Exception as e:
            print(f"distance_similarity计算错误: {e}")
            return 0

    def rank_features_with_direction(self, target_idx=-1, method='comprehensive'):
        """
        对所有特征列进行相似度排序（包含方向信息）
        """
        try:
            n_features = self.combined_data.shape[1] - 1
            rankings = []

            for i in range(n_features):
                if method == 'comprehensive':
                    score, detailed_scores, direction = self.comprehensive_similarity(i, target_idx)
                else:
                    # 简化版本，只使用相关系数方向
                    score = self.distance_similarity(i, target_idx)
                    direction = self.get_correlation_direction(i, target_idx)
                    detailed_scores = {}

                feature_name = self.get_feature_name(i)

                # 存储方向信息
                direction_symbol = '+' if direction > 0 else '-' if direction < 0 else '±'
                rankings.append((feature_name, score, direction_symbol, detailed_scores))

            # 按相似度绝对值降序排序
            rankings.sort(key=lambda x: abs(x[1]), reverse=True)
            return rankings

        except Exception as e:
            print(f"rank_features计算错误: {e}")
            return []

    def get_detailed_analysis(self, feature_idx, target_idx=-1):
        """
        获取特征的详细分析报告
        """
        try:
            # 基本相似度计算
            composite_score, scores, direction = self.comprehensive_similarity(feature_idx, target_idx)

            # 相关性方向
            corr_direction = self.get_correlation_direction(feature_idx, target_idx)
            direction_text = "正相关" if corr_direction > 0 else "负相关" if corr_direction < 0 else "相关性不显著"

            # PCA载荷分析
            pca_loadings = []
            for i in range(min(3, self.n_components)):
                feature_loading = self.components_[i, feature_idx]
                target_loading = self.components_[i, target_idx]
                pca_loadings.append({
                    'component': i + 1,
                    'feature_loading': feature_loading,
                    'target_loading': target_loading,
                    'variance_ratio': self.explained_variance_ratio_[i]
                })

            return {
                'feature_name': self.get_feature_name(feature_idx),
                'composite_score': composite_score,
                'direction': direction_text,
                'detailed_scores': scores,
                'pca_loadings': pca_loadings,
                'correlation_direction': corr_direction
            }

        except Exception as e:
            print(f"详细分析失败: {e}")
            return {}

    def get_pca_summary(self):
        """获取PCA模型摘要信息"""
        return {
            'n_components': self.n_components,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(
                self.explained_variance_ratio_) if self.explained_variance_ratio_ is not None else None,
            'components_shape': self.components_.shape if self.components_ is not None else None
        }


# 使用示例
def example_usage():
    # 生成示例数据
    np.random.seed(42)
    n_samples = 100

    # 创建特征数据
    features = np.random.randn(n_samples, 3)

    # 创建目标变量（与第一个特征正相关，与第二个特征负相关）
    target = features[:, 0] * 0.7 - features[:, 1] * 0.5 + np.random.randn(n_samples) * 0.2

    feature_names = ['特征A', '特征B', '特征C']

    # 创建分析器
    analyzer = PCASimilarity(n_components=2, feature_names=feature_names)
    analyzer.fit(features, target, target_name="目标变量")

    # 获取排序结果（包含方向）
    rankings = analyzer.rank_features_with_direction()

    print("特征相似度排名（包含方向）:")
    for i, (name, score, direction, detailed) in enumerate(rankings):
        print(f"{i + 1}. {name}: {score:.4f} ({direction})")

    # 获取详细分析
    print("\n详细分析报告:")
    for i in range(len(feature_names)):
        analysis = analyzer.get_detailed_analysis(i)
        if analysis:
            print(f"\n{analysis['feature_name']}:")
            print(f"  综合得分: {analysis['composite_score']:.4f}")
            print(f"  相关性: {analysis['direction']}")
            print(f"  详细得分: {analysis['detailed_scores']}")


if __name__ == "__main__":
    example_usage()