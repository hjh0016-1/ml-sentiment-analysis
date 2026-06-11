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

// ==================== 爬虫相关 ====================

/**
 * 爬虫配置
 */
export interface CrawlerConfig {
  source: 'mock' | 'file' | 'api';
  url?: string;
  maxItems: number;
  filters?: {
    minRating?: number;
    keywords?: string[];
    verifiedOnly?: boolean;
  };
}

/**
 * 爬虫状态
 */
export interface CrawlerStatus {
  status: 'idle' | 'running' | 'completed' | 'error';
  progress: number;
  collectedItems: number;
  errors: string[];
  startTime?: number;
  endTime?: number;
}

// ==================== 可视化相关 ====================

/**
 * 图表数据点
 */
export interface ChartDataPoint {
  x: number | string;
  y: number;
  label?: string;
  color?: string;
}

/**
 * 散点图数据
 */
export interface ScatterData {
  points: Array<{ x: number; y: number; cluster?: number; label?: string }>;
  xLabel: string;
  yLabel: string;
  title: string;
}

/**
 * 模型对比数据
 */
export interface ModelComparison {
  modelType: string;
  modelName: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    trainingTime: number;
    parameters?: number;
  };
  pros: string[];
  cons: string[];
  useCases: string[];
}