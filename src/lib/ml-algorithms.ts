/**
 * 传统机器学习算法实现
 * 包括 K-means 聚类和 PCA 降维
 */

import type {
  KMeansConfig,
  KMeansResult,
  ClusterStats,
  PCAConfig,
  PCAResult,
  FeatureVector,
} from '@/types';

/**
 * K-means 聚类算法实现
 */
export function kMeansClustering(
  data: FeatureVector[],
  config: KMeansConfig
): KMeansResult {
  const { k, maxIterations, tolerance } = config;

  if (data.length === 0 || data.length < k) {
    throw new Error('数据点数量必须大于聚类数量k');
  }

  const featureCount = data[0].features.length;
  const points = data.map((d) => d.features);

  // 初始化聚类中心（随机选择k个点）
  let centroids: number[][] = initializeCentroids(points, k);
  let clusters: number[] = [];
  let iterations = 0;
  let converged = false;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;

    // 分配点到最近的聚类中心
    clusters = assignClusters(points, centroids, config.distanceMetric);

    // 计算新的聚类中心
    const newCentroids = calculateNewCentroids(points, clusters, k, featureCount);

    // 检查是否收敛
    const movement = calculateCentroidMovement(centroids, newCentroids);
    if (movement < tolerance) {
      converged = true;
      centroids = newCentroids;
      break;
    }

    centroids = newCentroids;
  }

  // 计算聚类统计信息
  const clusterStats = calculateClusterStats(data, clusters, centroids);

  return {
    centroids,
    clusters,
    iterations,
    converged,
    clusterStats,
  };
}

/**
 * 初始化聚类中心
 */
function initializeCentroids(points: number[][], k: number): number[][] {
  const shuffled = [...points].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, k);
}

/**
 * 计算距离
 */
function calculateDistance(
  point1: number[],
  point2: number[],
  metric: 'euclidean' | 'cosine'
): number {
  if (metric === 'euclidean') {
    return Math.sqrt(
      point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0)
    );
  } else {
    // 余弦距离
    const dotProduct = point1.reduce((sum, val, i) => sum + val * point2[i], 0);
    const norm1 = Math.sqrt(point1.reduce((sum, val) => sum + val * val, 0));
    const norm2 = Math.sqrt(point2.reduce((sum, val) => sum + val * val, 0));
    return 1 - dotProduct / (norm1 * norm2);
  }
}

/**
 * 分配聚类
 */
function assignClusters(
  points: number[][],
  centroids: number[][],
  metric: 'euclidean' | 'cosine'
): number[] {
  return points.map((point) => {
    let minDistance = Infinity;
    let cluster = 0;

    centroids.forEach((centroid, index) => {
      const distance = calculateDistance(point, centroid, metric);
      if (distance < minDistance) {
        minDistance = distance;
        cluster = index;
      }
    });

    return cluster;
  });
}

/**
 * 计算新的聚类中心
 */
function calculateNewCentroids(
  points: number[][],
  clusters: number[],
  k: number,
  featureCount: number
): number[][] {
  const newCentroids: number[][] = [];

  for (let clusterId = 0; clusterId < k; clusterId++) {
    const clusterPoints = points.filter((_, i) => clusters[i] === clusterId);

    if (clusterPoints.length === 0) {
      // 如果聚类为空，保持原中心
      newCentroids.push(Array(featureCount).fill(0));
      continue;
    }

    // 计算平均值
    const centroid: number[] = [];
    for (let i = 0; i < featureCount; i++) {
      const avg =
        clusterPoints.reduce((sum, p) => sum + p[i], 0) / clusterPoints.length;
      centroid.push(avg);
    }
    newCentroids.push(centroid);
  }

  return newCentroids;
}

/**
 * 计算聚类中心移动距离
 */
function calculateCentroidMovement(
  oldCentroids: number[][],
  newCentroids: number[][]
): number {
  return oldCentroids.reduce((total, old, i) => {
    const movement = old.reduce((sum, val, j) => {
      return sum + Math.pow(val - newCentroids[i][j], 2);
    }, 0);
    return total + Math.sqrt(movement);
  }, 0);
}

/**
 * 计算聚类统计信息
 */
function calculateClusterStats(
  data: FeatureVector[],
  clusters: number[],
  centroids: number[][]
): ClusterStats[] {
  const k = centroids.length;
  const stats: ClusterStats[] = [];

  for (let clusterId = 0; clusterId < k; clusterId++) {
    const clusterData = data.filter((_, i) => clusters[i] === clusterId);
    const size = clusterData.length;

    // 计算平均特征值
    const avgFeatures = centroids[clusterId];

    // 计算情感分布
    const sentimentDistribution: Record<string, number> = {
      positive: clusterData.filter((d) => d.label === 2).length,
      neutral: clusterData.filter((d) => d.label === 1).length,
      negative: clusterData.filter((d) => d.label === 0).length,
    };

    stats.push({
      clusterId,
      size,
      avgFeatures,
      sentimentDistribution,
    });
  }

  return stats;
}

/**
 * PCA 降维算法实现
 */
export function performPCA(
  data: FeatureVector[],
  config: PCAConfig
): PCAResult {
  const { targetDimensions } = config;

  if (data.length === 0) {
    throw new Error('数据不能为空');
  }

  const points = data.map((d) => d.features);
  const featureCount = points[0].length;

  if (targetDimensions > featureCount) {
    throw new Error('目标维度不能超过原始特征维度');
  }

  // 步骤1：数据标准化（均值归零）
  const standardized = standardizeData(points);

  // 步骤2：计算协方差矩阵
  const covarianceMatrix = calculateCovarianceMatrix(standardized);

  // 步骤3：计算特征值和特征向量
  const { eigenvalues, eigenvectors } = computeEigen(covarianceMatrix);

  // 步骤4：选择主成分
  const sortedIndices = eigenvalues
    .map((_, i) => i)
    .sort((a, b) => eigenvalues[b] - eigenvalues[a]);

  const selectedIndices = sortedIndices.slice(0, targetDimensions);

  // 步骤5：构建投影矩阵
  const principalComponents = selectedIndices.map((i) => eigenvectors[i]);

  // 步骤6：计算解释方差
  const totalVariance = eigenvalues.reduce((sum, val) => sum + val, 0);
  const explainedVariance = selectedIndices.map(
    (i) => eigenvalues[i] / totalVariance
  );

  const cumulativeVariance: number[] = [];
  explainedVariance.forEach((variance, i) => {
    cumulativeVariance.push(
      variance + (i > 0 ? cumulativeVariance[i - 1] : 0)
    );
  });

  // 步骤7：投影数据
  const reducedData = projectData(standardized, principalComponents);

  return {
    reducedData,
    principalComponents,
    explainedVariance,
    cumulativeVariance,
  };
}

/**
 * 数据标准化（均值归零）
 */
function standardizeData(points: number[][]): number[][] {
  const featureCount = points[0].length;

  // 计算每个特征的均值
  const means: number[] = [];
  for (let i = 0; i < featureCount; i++) {
    const mean =
      points.reduce((sum, p) => sum + p[i], 0) / points.length;
    means.push(mean);
  }

  // 标准化
  return points.map((p) => p.map((val, i) => val - means[i]));
}

/**
 * 计算协方差矩阵
 */
function calculateCovarianceMatrix(points: number[][]): number[][] {
  const n = points.length;
  const featureCount = points[0].length;
  const covariance: number[][] = [];

  for (let i = 0; i < featureCount; i++) {
    const row: number[] = [];
    for (let j = 0; j < featureCount; j++) {
      const cov =
        points.reduce((sum, p) => sum + p[i] * p[j], 0) / (n - 1);
      row.push(cov);
    }
    covariance.push(row);
  }

  return covariance;
}

/**
 * 计算特征值和特征向量（简化实现：使用幂迭代法）
 */
function computeEigen(matrix: number[][]): {
  eigenvalues: number[];
  eigenvectors: number[][];
} {
  const n = matrix.length;
  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = [];

  // 使用幂迭代法计算前n个特征值
  for (let i = 0; i < n; i++) {
    let vector = Array(n).fill(0).map(() => Math.random());
    let eigenvalue = 0;

    // 幂迭代
    for (let iter = 0; iter < 100; iter++) {
      // 计算矩阵乘向量
      const newVector = matrix.map((row) =>
        row.reduce((sum, val, j) => sum + val * vector[j], 0)
      );

      // 计算范数
      const norm = Math.sqrt(
        newVector.reduce((sum, val) => sum + val * val, 0)
      );

      // 归一化
      vector = newVector.map((val) => val / (norm || 1));

      // 计算特征值（Rayleigh商）
      eigenvalue =
        vector.reduce((sum, v, j) => {
          return sum + v * matrix[j].reduce((s, val, k) => s + val * vector[k], 0);
        }, 0);
    }

    eigenvalues.push(Math.abs(eigenvalue));
    eigenvectors.push(vector);

    // deflate矩阵
    matrix = matrix.map((row, j) =>
      row.map((val, k) => val - eigenvalue * vector[j] * vector[k])
    );
  }

  return { eigenvalues, eigenvectors };
}

/**
 * 投影数据到主成分空间
 */
function projectData(
  data: number[][],
  principalComponents: number[][]
): number[][] {
  return data.map((point) => {
    return principalComponents.map((pc) =>
      point.reduce((sum, val, i) => sum + val * pc[i], 0)
    );
  });
}

/**
 * 计算轮廓系数（用于评估聚类质量）
 */
export function calculateSilhouetteScore(
  data: FeatureVector[],
  clusters: number[]
): number {
  if (data.length === 0 || clusters.length === 0) return 0;

  const points = data.map((d) => d.features);
  const scores: number[] = [];

  points.forEach((point, i) => {
    const cluster = clusters[i];

    // 计算同聚类平均距离（a）
    const sameClusterPoints = points.filter((_, j) => clusters[j] === cluster);
    const a =
      sameClusterPoints.length > 1
        ? sameClusterPoints
            .filter((_, j) => j !== i)
            .reduce((sum, p) => {
              return sum + calculateDistance(point, p, 'euclidean');
            }, 0) / (sameClusterPoints.length - 1)
        : 0;

    // 计算最近其他聚类的平均距离（b）
    const otherClusters = [...new Set(clusters)].filter((c) => c !== cluster);
    let minB = Infinity;

    otherClusters.forEach((otherCluster) => {
      const otherClusterPoints = points.filter(
        (_, j) => clusters[j] === otherCluster
      );
      const avgDistance =
        otherClusterPoints.reduce((sum, p) => {
          return sum + calculateDistance(point, p, 'euclidean');
        }, 0) / otherClusterPoints.length;

      if (avgDistance < minB) minB = avgDistance;
    });

    const b = minB === Infinity ? 0 : minB;

    // 计算轮廓系数
    const silhouette = Math.max(a, b) > 0 ? (b - a) / Math.max(a, b) : 0;
    scores.push(silhouette);
  });

  return scores.reduce((sum, s) => sum + s, 0) / scores.length;
}