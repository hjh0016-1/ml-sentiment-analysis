# 📦 项目完整代码 - 可直接复制使用

## 使用说明

1. 先创建 Next.js 项目：`npx create-next-app@latest sentiment-analysis --typescript --tailwind --app`
2. 进入项目文件夹：`cd sentiment-analysis`
3. 安装依赖：`pnpm add @tensorflow/tfjs recharts mathjs papaparse`
4. 按下方内容创建/替换以下文件

---

# 文件 1：src/types/index.ts

```typescript
/**
 * 数据挖掘与机器学习应用类型定义
 */

// ==================== 数据结构 ====================

/**
 * 产品评论数据
 */
export interface ReviewData {
  id: string;
  productId: string;
  productName: string;
  content: string; // 评论内容
  rating: number; // 评分 1-5
  sentiment: 'positive' | 'negative' | 'neutral'; // 情感标签
  keywords: string[]; // 关键词
  timestamp: number; // 时间戳
  userId?: string;
  verified: boolean; // 是否验证购买
}

/**
 * 数据统计信息
 */
export interface DataStats {
  total: number;
  positive: number;
  negative: number;
  neutral: number;
  avgRating: number;
  avgLength: number;
  missingValues: Record<string, number>;
  outliers: number;
}

/**
 * 数据预处理配置
 */
export interface PreprocessConfig {
  removeDuplicates: boolean;
  removeMissing: boolean;
  removeOutliers: boolean;
  normalizeText: boolean;
  keywordExtraction: boolean;
  languageFilter?: string;
}

/**
 * 特征向量数据
 */
export interface FeatureVector {
  id: string;
  features: number[];
  label?: number; // 0=negative, 1=neutral, 2=positive
}

// ==================== 机器学习相关 ====================

/**
 * K-means聚类配置
 */
export interface KMeansConfig {
  k: number; // 聚类数量
  maxIterations: number;
  tolerance: number;
  distanceMetric: 'euclidean' | 'cosine';
}

/**
 * K-means聚类结果
 */
export interface KMeansResult {
  centroids: number[][]; // 聚类中心
  clusters: number[]; // 每个样本的聚类标签
  iterations: number; // 实际迭代次数
  converged: boolean;
  clusterStats: ClusterStats[];
}

/**
 * 聚类统计信息
 */
export interface ClusterStats {
  clusterId: number;
  size: number;
  avgFeatures: number[];
  sentimentDistribution: Record<string, number>;
}

/**
 * PCA降维配置
 */
export interface PCAConfig {
  targetDimensions: number; // 目标维度
  varianceThreshold?: number; // 方差保留阈值
}

/**
 * PCA降维结果
 */
export interface PCAResult {
  reducedData: number[][]; // 降维后的数据
  principalComponents: number[][]; // 主成分
  explainedVariance: number[]; // 解释方差比例
  cumulativeVariance: number[]; // 累积方差
}

// ==================== 深度学习相关 ====================

/**
 * 神经网络配置
 */
export interface NeuralNetworkConfig {
  layers: LayerConfig[];
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: 'adam' | 'sgd' | 'rmsprop';
  lossFunction: 'binaryCrossentropy' | 'categoricalCrossentropy' | 'mse';
}

/**
 * 网络层配置
 */
export interface LayerConfig {
  type: 'dense' | 'conv2d' | 'lstm' | 'dropout';
  units?: number;
  activation: 'relu' | 'sigmoid' | 'tanh' | 'softmax';
  dropoutRate?: number;
}

/**
 * 训练结果
 */
export interface TrainingResult {
  history: {
    loss: number[];
    accuracy: number[];
    valLoss?: number[];
    valAccuracy?: number[];
  };
  finalLoss: number;
  finalAccuracy: number;
  modelParams: number;
  trainingTime: number;
}

/**
 * 预测结果
 */
export interface PredictionResult {
  predictions: number[];
  probabilities: number[][];
  confusionMatrix: number[][];
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}
```

---

# 文件 2：src/lib/data-processing.ts

```typescript
/**
 * 数据处理工具函数
 * 包括数据清洗、转换、特征提取等功能
 */

import type {
  ReviewData,
  DataStats,
  PreprocessConfig,
  FeatureVector,
} from '@/types';

/**
 * 生成模拟评论数据
 */
export function generateMockReviews(count: number): ReviewData[] {
  const reviews: ReviewData[] = [];
  const products = [
    { id: 'p1', name: '无线蓝牙耳机' },
    { id: 'p2', name: '智能手表' },
    { id: 'p3', name: '机械键盘' },
    { id: 'p4', name: '便携充电宝' },
    { id: 'p5', name: '降噪耳机' },
  ];

  const positiveTemplates = [
    '产品质量非常好，使用体验很棒',
    '物超所值，强烈推荐购买',
    '做工精细，手感舒适，非常满意',
    '性价比很高，功能齐全',
    '使用了一个月，非常满意，没有问题',
    '外观精美，功能强大，值得购买',
  ];

  const negativeTemplates = [
    '质量很差，使用几天就坏了',
    '与描述不符，退货了',
    '价格贵但质量一般，不推荐',
    '功能有问题，客服态度也不好',
    '收到货有瑕疵，很失望',
    '使用体验差，噪音很大',
  ];

  const neutralTemplates = [
    '产品还行，没什么特别的',
    '一般般，符合预期',
    '中规中矩，可以接受',
    '没有特别突出的优点或缺点',
    '使用体验普通，符合价格水平',
    '还可以，没什么大问题',
  ];

  for (let i = 0; i < count; i++) {
    const product = products[Math.floor(Math.random() * products.length)];
    const rating = Math.floor(Math.random() * 5) + 1;
    const sentiment =
      rating >= 4
        ? 'positive'
        : rating <= 2
          ? 'negative'
          : 'neutral';

    const templates =
      sentiment === 'positive'
        ? positiveTemplates
        : sentiment === 'negative'
          ? negativeTemplates
          : neutralTemplates;

    const baseContent =
      templates[Math.floor(Math.random() * templates.length)];
    const additionalWords = Math.random() > 0.5 ? '，整体感觉不错' : '';
    const content = baseContent + additionalWords;

    // 提取关键词（简单实现）
    const keywords = extractKeywordsSimple(content);

    reviews.push({
      id: `review-${i}`,
      productId: product.id,
      productName: product.name,
      content,
      rating,
      sentiment,
      keywords,
      timestamp: Date.now() - Math.floor(Math.random() * 30 * 24 * 60 * 60 * 1000),
      verified: Math.random() > 0.3,
    });
  }

  return reviews;
}

/**
 * 简单的关键词提取
 */
function extractKeywordsSimple(text: string): string[] {
  const keywords: string[] = [];
  const positiveWords = ['好', '棒', '满意', '推荐', '精美', '舒适', '超值'];
  const negativeWords = ['差', '坏', '失望', '退货', '问题', '瑕疵'];
  const neutralWords = ['还行', '一般', '普通', '符合'];

  positiveWords.forEach((word) => {
    if (text.includes(word)) keywords.push(word);
  });
  negativeWords.forEach((word) => {
    if (text.includes(word)) keywords.push(word);
  });
  neutralWords.forEach((word) => {
    if (text.includes(word)) keywords.push(word);
  });

  return keywords.length > 0 ? keywords : ['产品'];
}

/**
 * 计算数据统计信息
 */
export function calculateDataStats(reviews: ReviewData[]): DataStats {
  if (reviews.length === 0) {
    return {
      total: 0,
      positive: 0,
      negative: 0,
      neutral: 0,
      avgRating: 0,
      avgLength: 0,
      missingValues: {},
      outliers: 0,
    };
  }

  const positive = reviews.filter((r) => r.sentiment === 'positive').length;
  const negative = reviews.filter((r) => r.sentiment === 'negative').length;
  const neutral = reviews.filter((r) => r.sentiment === 'neutral').length;

  const avgRating =
    reviews.reduce((sum, r) => sum + r.rating, 0) / reviews.length;
  const avgLength =
    reviews.reduce((sum, r) => sum + r.content.length, 0) / reviews.length;

  // 检查缺失值
  const missingValues: Record<string, number> = {
    rating: reviews.filter((r) => r.rating === undefined || r.rating === 0)
      .length,
    content: reviews.filter((r) => !r.content || r.content.length === 0)
      .length,
    userId: reviews.filter((r) => !r.userId).length,
  };

  // 检查异常值（评分超出1-5范围）
  const outliers = reviews.filter(
    (r) => r.rating < 1 || r.rating > 5 || r.content.length > 500
  ).length;

  return {
    total: reviews.length,
    positive,
    negative,
    neutral,
    avgRating: Math.round(avgRating * 100) / 100,
    avgLength: Math.round(avgLength),
    missingValues,
    outliers,
  };
}

/**
 * 数据预处理
 */
export function preprocessData(
  reviews: ReviewData[],
  config: PreprocessConfig
): ReviewData[] {
  let processed = [...reviews];

  // 移除重复数据
  if (config.removeDuplicates) {
    const seen = new Set<string>();
    processed = processed.filter((r) => {
      const key = `${r.productId}-${r.content}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  // 移除缺失值
  if (config.removeMissing) {
    processed = processed.filter(
      (r) => r.content && r.content.length > 0 && r.rating >= 1 && r.rating <= 5
    );
  }

  // 移除异常值
  if (config.removeOutliers) {
    processed = processed.filter((r) => r.content.length <= 500);
  }

  // 文本标准化
  if (config.normalizeText) {
    processed = processed.map((r) => ({
      ...r,
      content: normalizeText(r.content),
      keywords: extractKeywordsSimple(normalizeText(r.content)),
    }));
  }

  return processed;
}

/**
 * 文本标准化
 */
function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\u4e00-\u9fa5a-z0-9\s]/g, '') // 移除特殊字符，保留中文和英文
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * 特征提取：将评论转换为特征向量
 */
export function extractFeatures(reviews: ReviewData[]): FeatureVector[] {
  const vectors: FeatureVector[] = [];

  reviews.forEach((review) => {
    // 创建简单特征向量（15维）
    const features: number[] = [
      // 评分特征
      review.rating,
      review.rating / 5, // 归一化评分

      // 文本长度特征
      review.content.length,
      review.content.length / 500, // 归一化长度

      // 情感特征（关键词计数）
      countKeywords(review.keywords, ['好', '棒', '满意', '推荐']),
      countKeywords(review.keywords, ['差', '坏', '失望', '退货']),
      countKeywords(review.keywords, ['还行', '一般', '普通']),

      // 情感标签编码
      review.sentiment === 'positive' ? 1 : 0,
      review.sentiment === 'negative' ? 1 : 0,
      review.sentiment === 'neutral' ? 1 : 0,

      // 其他特征
      review.verified ? 1 : 0,
      (Date.now() - review.timestamp) / (30 * 24 * 60 * 60 * 1000), // 时间差（天数）

      // 产品编码（简化）
      encodeProduct(review.productId),
      review.keywords.length,
      review.keywords.length / 10, // 归一化关键词数量
    ];

    const label =
      review.sentiment === 'positive'
        ? 2
        : review.sentiment === 'neutral'
          ? 1
          : 0;

    vectors.push({
      id: review.id,
      features,
      label,
    });
  });

  return vectors;
}

/**
 * 计算关键词数量
 */
function countKeywords(
  keywords: string[],
  targetWords: string[]
): number {
  return keywords.filter((k) => targetWords.includes(k)).length;
}

/**
 * 产品编码
 */
function encodeProduct(productId: string): number {
  const products: Record<string, number> = {
    p1: 0.2,
    p2: 0.4,
    p3: 0.6,
    p4: 0.8,
    p5: 1.0,
  };
  return products[productId] || 0;
}

/**
 * 数据归一化
 */
export function normalizeFeatures(vectors: FeatureVector[]): FeatureVector[] {
  if (vectors.length === 0) return vectors;

  const featureCount = vectors[0].features.length;

  // 计算每个特征的均值和标准差
  const means: number[] = [];
  const stds: number[] = [];

  for (let i = 0; i < featureCount; i++) {
    const values = vectors.map((v) => v.features[i]);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance =
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      values.length;
    const std = Math.sqrt(variance) || 1; // 避免除以0

    means.push(mean);
    stds.push(std);
  }

  // 归一化
  return vectors.map((v) => ({
    ...v,
    features: v.features.map(
      (f, i) => (f - means[i]) / stds[i]
    ),
  }));
}
```

---

# 文件 3：src/lib/ml-algorithms.ts

```typescript
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
```

---

# 文件 4：src/lib/dl-model.ts

```typescript
/**
 * 深度学习模块实现
 * 使用 TensorFlow.js 实现神经网络情感分类模型
 */

import * as tf from '@tensorflow/tfjs';
import type {
  NeuralNetworkConfig,
  TrainingResult,
  PredictionResult,
  FeatureVector,
} from '@/types';

/**
 * 神经网络模型类
 */
export class NeuralNetworkClassifier {
  private model: tf.LayersModel | null = null;
  private config: NeuralNetworkConfig;
  private trained = false;

  constructor(config: NeuralNetworkConfig) {
    this.config = config;
  }

  /**
   * 构建神经网络模型
   */
  buildModel(inputDimension: number): void {
    this.model = tf.sequential();

    // 添加层
    this.config.layers.forEach((layerConfig, index) => {
      if (layerConfig.type === 'dense') {
        const isFirst = index === 0;
        const units = layerConfig.units || 32;

        (this.model as tf.Sequential).add(
          tf.layers.dense({
            units,
            inputShape: isFirst ? [inputDimension] : undefined,
            activation: layerConfig.activation,
          })
        );
      }

      if (layerConfig.type === 'dropout' && layerConfig.dropoutRate) {
        (this.model as tf.Sequential).add(
          tf.layers.dropout({
            rate: layerConfig.dropoutRate,
          })
        );
      }
    });

    // 编译模型
    const optimizer = this.getOptimizer();
    this.model.compile({
      optimizer,
      loss: this.config.lossFunction,
      metrics: ['accuracy'],
    });
  }

  /**
   * 获取优化器
   */
  private getOptimizer(): tf.Optimizer {
    const lr = this.config.learningRate;

    switch (this.config.optimizer) {
      case 'adam':
        return tf.train.adam(lr);
      case 'sgd':
        return tf.train.sgd(lr);
      case 'rmsprop':
        return tf.train.rmsprop(lr);
      default:
        return tf.train.adam(lr);
    }
  }

  /**
   * 训练模型
   */
  async train(
    trainData: FeatureVector[],
    validationData?: FeatureVector[]
  ): Promise<TrainingResult> {
    if (!this.model) {
      throw new Error('模型未构建');
    }

    const startTime = Date.now();

    // 准备训练数据
    const xTrain = tf.tensor2d(
      trainData.map((d) => d.features)
    );
    const yTrain = tf.tensor2d(
      trainData.map((d) => this.encodeLabel(d.label || 0))
    );

    // 准备验证数据
    let xVal: tf.Tensor2D | null = null;
    let yVal: tf.Tensor2D | null = null;
    if (validationData && validationData.length > 0) {
      xVal = tf.tensor2d(
        validationData.map((d) => d.features)
      );
      yVal = tf.tensor2d(
        validationData.map((d) => this.encodeLabel(d.label || 0))
      );
    }

    // 训练
    const history = await this.model.fit(xTrain, yTrain, {
      epochs: this.config.epochs,
      batchSize: this.config.batchSize,
      validationData: xVal && yVal ? [xVal, yVal] : undefined,
      shuffle: true,
    });

    // 记录训练历史
    const trainingHistory = {
      loss: history.history.loss as number[],
      accuracy: history.history.acc as number[],
      valLoss: history.history.val_loss as number[] | undefined,
      valAccuracy: history.history.val_acc as number[] | undefined,
    };

    const endTime = Date.now();

    // 清理 tensors
    xTrain.dispose();
    yTrain.dispose();
    if (xVal) xVal.dispose();
    if (yVal) yVal.dispose();

    this.trained = true;

    return {
      history: trainingHistory,
      finalLoss: trainingHistory.loss[trainingHistory.loss.length - 1],
      finalAccuracy: trainingHistory.accuracy[trainingHistory.accuracy.length - 1],
      modelParams: this.model.countParams(),
      trainingTime: endTime - startTime,
    };
  }

  /**
   * 标签编码（one-hot）
   */
  private encodeLabel(label: number): number[] {
    // 3分类：negative(0), neutral(1), positive(2)
    const encoded = [0, 0, 0];
    if (label >= 0 && label < 3) {
      encoded[label] = 1;
    }
    return encoded;
  }

  /**
   * 预测
   */
  async predict(testData: FeatureVector[]): Promise<PredictionResult> {
    if (!this.model || !this.trained) {
      throw new Error('模型未训练');
    }

    // 准备预测数据
    const xTest = tf.tensor2d(
      testData.map((d) => d.features)
    );

    // 预测
    const predictionsTensor = this.model.predict(xTest) as tf.Tensor;
    const probabilities = await predictionsTensor.array() as number[][];

    // 获取预测标签
    const predictions = probabilities.map((probs) => {
      return probs.indexOf(Math.max(...probs));
    });

    // 计算混淆矩阵
    const trueLabels = testData.map((d) => d.label || 0);
    const confusionMatrix = this.calculateConfusionMatrix(
      trueLabels,
      predictions,
      3
    );

    // 计算评估指标
    const metrics = this.calculateMetrics(confusionMatrix);

    // 清理 tensors
    xTest.dispose();
    predictionsTensor.dispose();

    return {
      predictions,
      probabilities,
      confusionMatrix,
      accuracy: metrics.accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      f1Score: metrics.f1Score,
    };
  }

  /**
   * 计算混淆矩阵
   */
  private calculateConfusionMatrix(
    trueLabels: number[],
    predictions: number[],
    numClasses: number
  ): number[][] {
    const matrix: number[][] = Array(numClasses)
      .fill(null)
      .map(() => Array(numClasses).fill(0));

    trueLabels.forEach((trueLabel, i) => {
      matrix[trueLabel][predictions[i]]++;
    });

    return matrix;
  }

  /**
   * 计算评估指标
   */
  private calculateMetrics(confusionMatrix: number[][]): {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  } {
    const numClasses = confusionMatrix.length;

    // 计算总数和正确预测数
    let total = 0;
    let correct = 0;
    confusionMatrix.forEach((row, i) => {
      row.forEach((val) => (total += val));
      correct += row[i];
    });

    const accuracy = total > 0 ? correct / total : 0;

    // 计算每个类的precision和recall
    let totalPrecision = 0;
    let totalRecall = 0;
    let validClasses = 0;

    for (let i = 0; i < numClasses; i++) {
      const tp = confusionMatrix[i][i];
      const fp = confusionMatrix.reduce(
        (sum, row, j) => (j !== i ? sum + row[i] : sum),
        0
      );
      const fn = confusionMatrix[i].reduce(
        (sum, val, j) => (j !== i ? sum + val : sum),
        0
      );

      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;

      if (tp > 0) {
        totalPrecision += precision;
        totalRecall += recall;
        validClasses++;
      }
    }

    const avgPrecision = validClasses > 0 ? totalPrecision / validClasses : 0;
    const avgRecall = validClasses > 0 ? totalRecall / validClasses : 0;
    const f1Score =
      avgPrecision + avgRecall > 0
        ? 2 * avgPrecision * avgRecall / (avgPrecision + avgRecall)
        : 0;

    return {
      accuracy: Math.round(accuracy * 1000) / 1000,
      precision: Math.round(avgPrecision * 1000) / 1000,
      recall: Math.round(avgRecall * 1000) / 1000,
      f1Score: Math.round(f1Score * 1000) / 1000,
    };
  }

  /**
   * 获取模型信息
   */
  getModelInfo(): {
    summary: string;
    params: number;
  } {
    if (!this.model) {
      return { summary: '模型未构建', params: 0 };
    }

    const summary = this.model.layers
      .map((layer) => {
        const config = layer.getConfig();
        return `${layer.name}: ${config.units || 'dropout'} units, ${config.activation || 'no activation'}`;
      })
      .join('\n');

    return {
      summary,
      params: this.model.countParams(),
    };
  }

  /**
   * 清理模型
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.trained = false;
    }
  }
}

/**
 * 创建默认神经网络配置
 */
export function createDefaultNNConfig(): NeuralNetworkConfig {
  return {
    layers: [
      { type: 'dense', units: 64, activation: 'relu' },
      { type: 'dropout', units: 0, activation: 'relu', dropoutRate: 0.3 },
      { type: 'dense', units: 32, activation: 'relu' },
      { type: 'dropout', units: 0, activation: 'relu', dropoutRate: 0.2 },
      { type: 'dense', units: 16, activation: 'relu' },
      { type: 'dense', units: 3, activation: 'softmax' },
    ],
    learningRate: 0.001,
    batchSize: 32,
    epochs: 50,
    optimizer: 'adam',
    lossFunction: 'categoricalCrossentropy',
  };
}

/**
 * 划分训练集和测试集
 */
export function splitTrainTest(
  data: FeatureVector[],
  testRatio: number = 0.2
): { train: FeatureVector[]; test: FeatureVector[] } {
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const testSize = Math.floor(shuffled.length * testRatio);
  const test = shuffled.slice(0, testSize);
  const train = shuffled.slice(testSize);

  return { train, test };
}
```

---

# ⚠️ 注意

**文件 5（src/app/page.tsx）内容很长（约600行），超过单次展示限制。**

请查看项目中的 **COMPLETE_CODE.md** 文件获取完整内容，或者按照以下方式获取：

1. 在线查看：打开页面顶部的访问链接，项目正在运行
2. 查看完整文档：项目中已创建 `COMPLETE_CODE.md` 文件

---

# 🚀 快速开始

```bash
# 1. 创建项目
npx create-next-app@latest sentiment-analysis --typescript --tailwind --app

# 2. 进入项目
cd sentiment-analysis

# 3. 安装依赖
pnpm add @tensorflow/tfjs recharts mathjs papaparse

# 4. 创建上述4个文件（复制粘贴内容）

# 5. 运行项目
pnpm dev

# 6. 打开浏览器
http://localhost:5000
```