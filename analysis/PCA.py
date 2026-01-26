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

    def __init__(self, n_components=2, standardize=True):
        self.n_components = n_components
        self.standardize = standardize
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() if standardize else None
        self.explained_variance_ratio_ = None
        self.components_ = None

    def fit(self, data_columns, target_column):
        """
        拟合PCA模型到数据
        """
        if isinstance(data_columns, pd.DataFrame):
            data_columns = data_columns.values
        if isinstance(target_column, pd.Series):
            target_column = target_column.values

        # 将目标列作为额外特征合并
        target_reshaped = target_column.reshape(-1, 1)
        self.combined_data = np.hstack([data_columns, target_reshaped])

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

    def explained_variance_similarity(self, feature_idx, target_idx=-1):
        """
        基于方差解释比重的加权相似度（修复版本）
        """
        try:
            # 获取主成分上的投影
            feature_proj = self.transformed_data[:, feature_idx] if self.n_components > 1 else self.transformed_data
            target_proj = self.transformed_data[:, target_idx] if self.n_components > 1 else self.transformed_data

            # 确保数组维度正确
            if self.n_components == 1:
                # 一维情况：直接计算相关系数
                feature_vals = feature_proj.flatten()
                target_vals = target_proj.flatten()
                corr, _ = pearsonr(feature_vals, target_vals)
                weighted_similarity = abs(corr) * self.explained_variance_ratio_[0]
                return weighted_similarity
            else:
                # 多维情况：各维度加权计算
                weighted_similarity = 0
                total_variance = 0

                for i in range(self.n_components):
                    # 正确处理二维索引
                    feature_vals = feature_proj[:, i] if feature_proj.ndim > 1 else feature_proj
                    target_vals = target_proj[:, i] if target_proj.ndim > 1 else target_proj

                    corr, _ = pearsonr(feature_vals, target_vals)
                    weight = self.explained_variance_ratio_[i]
                    weighted_similarity += abs(corr) * weight
                    total_variance += weight

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
        对所有特征列进行相似度排序
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

                rankings.append((i, score))

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