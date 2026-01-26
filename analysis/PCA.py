import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr


class PCASimilarity:
    """
    基于PCA的多数据列与目标数据列相似度分析类
    """

    def __init__(self, n_components=2, standardize=True, feature_names=None):
        self.n_components = n_components
        self.standardize = standardize
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() if standardize else None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.feature_names = feature_names  # 新增特征名称参数

    def fit(self, data_columns, target_column, target_name="target"):
        """
        拟合PCA模型到数据
        """
        if isinstance(data_columns, pd.DataFrame):
            # 如果输入是DataFrame，自动获取特征名称
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
            self.feature_names = [f'{i}' for i in range(n_features)]

        # 添加目标名称到特征名称列表
        self.all_feature_names = self.feature_names + [self.target_name]

        # 改进：特征和目标分别标准化
        if self.standardize:
            data_scaled = self.scaler.fit_transform(data_columns)
            target_scaled = (target_column - target_column.mean()) / target_column.std()
            self.combined_data_scaled = np.hstack([data_scaled, target_scaled.reshape(-1, 1)])
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
        return f"{feature_idx}"

    def explained_variance_similarity(self, feature_idx, target_idx=-1):
        """
        基于方差解释比重的加权相似度（正确版本）
        """
        try:
            if self.transformed_data is None or self.explained_variance_ratio_ is None:
                return 0

            # 正确的逻辑：比较特征列和目标列在PCA空间中的行为
            weighted_similarity = 0
            total_variance = 0

            for i in range(self.n_components):
                try:
                    # 关键修复：获取特征和目标在PCA载荷向量上的"相关性"
                    # 使用PCA载荷（components_）而不是投影值（transformed_data）
                    feature_loading = self.components_[i, feature_idx]  # 特征在第i主成分的载荷
                    target_loading = self.components_[i, target_idx]  # 目标在第i主成分的载荷

                    # 计算载荷向量的相似度（使用余弦相似度的思想）
                    loading_similarity = abs(feature_loading * target_loading)

                    # 用方差解释比重加权
                    weight = self.explained_variance_ratio_[i]
                    weighted_similarity += loading_similarity * weight
                    total_variance += weight

                except Exception as dim_error:
                    print(f"维度 {i} 计算失败: {dim_error}")
                    continue

            return weighted_similarity / total_variance if total_variance > 0 else 0

        except Exception as e:
            print(f"explained_variance_similarity计算错误: {e}")
            return 0

    def cosine_similarity_pca(self, feature_idx, target_idx=-1):
        """
        在PCA空间计算特征与目标之间的余弦相似度
        """
        try:
            # 获取PCA载荷向量
            feature_pca_loading = self.components_[:, feature_idx]
            target_pca_loading = self.components_[:, target_idx]

            # 计算余弦相似度
            cosine_sim = 1 - cosine(feature_pca_loading, target_pca_loading)
            return max(0, cosine_sim)
        except Exception as e:
            print(f"cosine_similarity_pca计算错误: {e}")
            return 0

    def distance_similarity(self, feature_idx, target_idx=-1, metric='euclidean'):
        """
        基于距离的相似度计算
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

    def correlation_similarity(self, feature_idx, target_idx=-1):
        """
        基于相关系数的相似度计算
        """
        try:
            feature_data = self.combined_data[:, feature_idx]
            target_data = self.combined_data[:, target_idx]

            corr, _ = pearsonr(feature_data, target_data)
            return abs(corr)
        except Exception as e:
            print(f"correlation_similarity计算错误: {e}")
            return 0

    def comprehensive_similarity(self, feature_idx, target_idx=-1, weights=None):
        """
        综合相似度评估（修复版本）
        """
        if weights is None:
            weights = {
                'pca_cosine': 0.3,
                'variance_weighted': 0.3,
                'distance': 0.2,
                'correlation': 0.2
            }

        try:
            # 计算各指标得分
            scores = {}
            scores['pca_cosine'] = self.cosine_similarity_pca(feature_idx, target_idx)
            scores['variance_weighted'] = self.explained_variance_similarity(feature_idx, target_idx)
            scores['distance'] = self.distance_similarity(feature_idx, target_idx)
            scores['correlation'] = self.correlation_similarity(feature_idx, target_idx)

            # 计算加权综合得分
            composite_score = 0
            for key, weight in weights.items():
                composite_score += scores[key] * weight

            return composite_score, scores

        except Exception as e:
            print(f"comprehensive_similarity计算错误: {e}")
            return 0, {}

    def rank_features(self, target_idx=-1, method='comprehensive'):
        """
        对所有特征列进行相似度排序（使用特征名称）
        """
        try:
            n_features = self.combined_data.shape[1] - 1  # 排除目标列
            rankings = []

            for i in range(n_features):
                if method == 'comprehensive':
                    score, detailed_scores = self.comprehensive_similarity(i, target_idx)
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

    def get_pca_summary(self):
        """获取PCA模型摘要信息"""
        return {
            'n_components': self.n_components,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(
                self.explained_variance_ratio_) if self.explained_variance_ratio_ is not None else None,
            'components_shape': self.components_.shape if self.components_ is not None else None
        }