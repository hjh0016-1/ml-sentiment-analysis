# 本地运行指南 - 数据挖掘与机器学习实践平台

## 环境要求

- Node.js 18+ 或 24（推荐）
- pnpm（必须使用，不能用npm）

## 创建步骤

### 第一步：创建项目

```bash
npx create-next-app@latest ml-sentiment-analysis --typescript --tailwind --app --src-dir
cd ml-sentiment-analysis
pnpm add @tensorflow/tfjs recharts mathjs papaparse
```

### 第二步：创建文件结构

你需要创建以下文件夹和文件：

```
src/
├── types/
│   └── index.ts          （类型定义）
├── lib/
│   ├── data-processing.ts  （数据处理）
│   ├── ml-algorithms.ts    （机器学习算法）
│   ├── dl-model.ts         （深度学习模型）
│   └── utils.ts            （工具函数，已有）
├── components/
│   └── ui/                 （已有UI组件）
└── app/
    ├── layout.tsx          （已有）
    ├── globals.css         （已有）
    └── page.tsx            （主页面，需要替换）
```

---

## 核心代码文件

### 文件1: src/types/index.ts

创建 `src/types/index.ts` 文件，复制以下完整内容：

```typescript
// ==================== src/types/index.ts ====================
// 数据挖掘与机器学习实践平台 - 类型定义

// 原始评论数据
export interface Review {
  id: string;
  productId: string;
  productName: string;
  rating: number; // 1-5
  content: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  keywords: string[];
  timestamp: number;
  userId?: string;
}

// 预处理后的数据
export interface ProcessedReview {
  id: string;
  rating: number;
  sentiment: 'positive' | 'negative' | 'neutral';
  contentLength: number;
  featureVector: number[];
  keywords: string[];
}

// 数据统计信息
export interface DataStats {
  total: number;
  positive: number;
  negative: number;
  neutral: number;
  avgRating: number;
  avgContentLength: number;
  ratingDistribution: { [key: number]: number };
}

// 预处理配置
export interface PreprocessConfig {
  removeDuplicates: boolean;
  removeMissing: boolean;
  removeOutliers: boolean;
  normalizeText: boolean;
  keywordExtraction: boolean;
  languageFilter?: string;
}

// K-means聚类结果
export interface KMeansResult {
  centroids: number[][];
  clusters: number[];
  iterations: number;
  converged: boolean;
  clusterStats: {
    size: number;
    avgRating: number;
    dominantSentiment: string;
  }[];
}

// PCA降维结果
export interface PCAResult {
  reducedData: number[][];
  components: number[][];
  explainedVariance: number[];
  cumulativeVariance: number[];
}

// 神经网络配置
export interface NeuralNetworkConfig {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  learningRate: number;
  epochs: number;
  batchSize: number;
  dropoutRate: number;
}

// 训练历史
export interface TrainingHistory {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss?: number;
  valAccuracy?: number;
}

// 模型预测结果
export interface PredictionResult {
  predictions: number[];
  probabilities: number[][];
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
}

// 深度学习结果
export interface DLResult {
  model: any;
  history: TrainingHistory[];
  prediction?: PredictionResult;
}

// 爬虫配置
export interface CrawlerConfig {
  targetSites: string[];
  maxReviews: number;
  delay: number;
  retries: number;
}

// 爬虫进度
export interface CrawlerProgress {
  current: number;
  total: number;
  status: 'idle' | 'running' | 'completed' | 'error';
  message: string;
}

// 模型对比结果
export interface ModelComparison {
  method: string;
  accuracy?: number;
  silhouetteScore?: number;
  explainedVariance?: number;
  f1Score?: number;
  trainingTime?: number;
  pros: string[];
  cons: string[];
}
```

---

### 文件2: src/lib/data-processing.ts

创建 `src/lib/data-processing.ts` 文件，复制以下完整内容：

```typescript
// ==================== src/lib/data-processing.ts ====================
// 数据处理工具 - 数据采集、预处理、特征提取

import { Review, ProcessedReview, DataStats, PreprocessConfig } from '@/types';

// 生成模拟评论数据
export function generateMockReviews(count: number = 200): Review[] {
  const reviews: Review[] = [];
  const products = ['智能手机', '笔记本电脑', '耳机', '相机', '平板电脑'];
  const positiveKeywords = ['好', '优秀', '满意', '推荐', '质量好', '性价比高', '速度快', '好用'];
  const negativeKeywords = ['差', '失望', '退货', '不推荐', '质量差', '慢', '卡顿', '问题'];
  const neutralKeywords = ['一般', '还行', '普通', '凑合', '中等', '正常'];
  
  const positiveTemplates = [
    '产品非常{keyword}，{keyword2}，非常满意这次购物体验',
    '用了几天，感觉{keyword}，{keyword2}，推荐购买',
    '质量{keyword}，性能{keyword2}，性价比很高',
    '收到货很惊喜，{keyword}，{keyword2}，下次还会买',
  ];
  
  const negativeTemplates = [
    '产品{keyword}，{keyword2}，打算退货',
    '用了不到一周就{keyword}了，{keyword2}，很失望',
    '质量太{keyword}，{keyword2}，不推荐',
    '收到货发现{keyword}，{keyword2}，失望',
  ];
  
  const neutralTemplates = [
    '产品{keyword}，{keyword2}，没什么特别的',
    '用着{keyword}，性能{keyword2}，还行吧',
    '质量{keyword}，{keyword2}，普通水平',
    '收到货感觉{keyword}，{keyword2}，凑合用',
  ];
  
  for (let i = 0; i < count; i++) {
    const sentiment = Math.random() < 0.35 ? 'positive' : 
                      Math.random() < 0.65 ? 'negative' : 'neutral';
    
    let template, keywords: string[];
    if (sentiment === 'positive') {
      template = positiveTemplates[Math.floor(Math.random() * positiveTemplates.length)];
      keywords = positiveKeywords.slice(0, 2 + Math.floor(Math.random() * 3));
    } else if (sentiment === 'negative') {
      template = negativeTemplates[Math.floor(Math.random() * negativeTemplates.length)];
      keywords = negativeKeywords.slice(0, 2 + Math.floor(Math.random() * 3));
    } else {
      template = neutralTemplates[Math.floor(Math.random() * neutralTemplates.length)];
      keywords = neutralKeywords.slice(0, 2 + Math.floor(Math.random() * 3));
    }
    
    const rating = sentiment === 'positive' ? 4 + Math.floor(Math.random() * 2) :
                   sentiment === 'negative' ? 1 + Math.floor(Math.random() * 2) :
                   3;
    
    const content = template
      .replace('{keyword}', keywords[0] || '一般')
      .replace('{keyword2}', keywords[1] || '还行');
    
    reviews.push({
      id: `review-${i + 1}`,
      productId: `product-${Math.floor(Math.random() * products.length) + 1}`,
      productName: products[Math.floor(Math.random() * products.length)],
      rating,
      content: content + '。' + (Math.random() > 0.5 ? '总体来说比较满意。' : ''),
      sentiment,
      keywords,
      timestamp: Date.now() - Math.floor(Math.random() * 30 * 24 * 60 * 60 * 1000),
      userId: `user-${Math.floor(Math.random() * 100) + 1}`,
    });
  }
  
  return reviews;
}

// 计算数据统计信息
export function calculateDataStats(reviews: Review[]): DataStats {
  const stats: DataStats = {
    total: reviews.length,
    positive: 0,
    negative: 0,
    neutral: 0,
    avgRating: 0,
    avgContentLength: 0,
    ratingDistribution: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
  };
  
  let totalRating = 0;
  let totalLength = 0;
  
  for (const review of reviews) {
    if (review.sentiment === 'positive') stats.positive++;
    else if (review.sentiment === 'negative') stats.negative++;
    else stats.neutral++;
    
    totalRating += review.rating;
    totalLength += review.content.length;
    stats.ratingDistribution[review.rating]++;
  }
  
  stats.avgRating = totalRating / reviews.length;
  stats.avgContentLength = totalLength / reviews.length;
  
  return stats;
}

// 预处理数据
export function preprocessData(
  reviews: Review[],
  config: PreprocessConfig
): ProcessedReview[] {
  let processedReviews = [...reviews];
  
  // 移除重复数据
  if (config.removeDuplicates) {
    const seen = new Set<string>();
    processedReviews = processedReviews.filter(review => {
      const key = review.content;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }
  
  // 移除缺失值
  if (config.removeMissing) {
    processedReviews = processedReviews.filter(review => 
      review.content && review.content.length > 0 &&
      review.rating >= 1 && review.rating <= 5
    );
  }
  
  // 移除异常值
  if (config.removeOutliers) {
    processedReviews = processedReviews.filter(review =>
      review.content.length >= 5 && review.content.length <= 500
    );
  }
  
  // 提取特征
  return processedReviews.map(review => ({
    id: review.id,
    rating: review.rating,
    sentiment: review.sentiment,
    contentLength: review.content.length,
    featureVector: extractFeatureVector(review),
    keywords: review.keywords,
  }));
}

// 提取15维特征向量
export function extractFeatureVector(review: Review): number[] {
  const features: number[] = [];
  
  // 1. 评分（归一化到0-1）
  features.push(review.rating / 5);
  
  // 2. 评论长度（归一化）
  features.push(Math.min(review.content.length / 200, 1));
  
  // 3-5. 情感特征（3个二进制）
  features.push(review.sentiment === 'positive' ? 1 : 0);
  features.push(review.sentiment === 'negative' ? 1 : 0);
  features.push(review.sentiment === 'neutral' ? 1 : 0);
  
  // 6-10. 关键词计数（5个常见情感词）
  const positiveWords = ['好', '优秀', '满意', '推荐', '快'];
  const negativeWords = ['差', '失望', '退货', '问题', '慢'];
  
  let positiveCount = 0;
  let negativeCount = 0;
  
  for (const keyword of review.keywords) {
    if (positiveWords.includes(keyword)) positiveCount++;
    if (negativeWords.includes(keyword)) negativeCount++;
  }
  
  features.push(Math.min(positiveCount / 5, 1));
  features.push(Math.min(negativeCount / 5, 1));
  features.push(review.keywords.length / 10);
  features.push(review.content.includes('推荐') ? 1 : 0);
  features.push(review.content.includes('退货') ? 1 : 0);
  
  // 11-15. 其他特征
  features.push(review.rating >= 4 ? 1 : 0);
  features.push(review.rating <= 2 ? 1 : 0);
  features.push(review.content.includes('！') ? 1 : 0);
  features.push(review.timestamp % 2 === 0 ? 0.5 : 0.3);
  features.push(Math.random() * 0.2 + 0.4); // 随机噪声
  
  return features;
}

// 归一化特征向量（Z-score标准化）
export function normalizeFeatures(features: number[][]): number[][] {
  if (features.length === 0) return features;
  
  const featureCount = features[0].length;
  const means: number[] = [];
  const stds: number[] = [];
  
  // 计算每个特征的均值和标准差
  for (let j = 0; j < featureCount; j++) {
    const values = features.map(f => f[j]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance) || 1;
    means.push(mean);
    stds.push(std);
  }
  
  // 应用Z-score标准化
  return features.map(feature =>
    feature.map((value, j) => (value - means[j]) / stds[j])
  );
}
```

---

### 文件3: src/lib/ml-algorithms.ts

创建 `src/lib/ml-algorithms.ts` 文件，复制以下完整内容：

```typescript
// ==================== src/lib/ml-algorithms.ts ====================
// 传统机器学习算法 - K-means聚类、PCA降维

import { KMeansResult, PCAResult } from '@/types';

// K-means聚类算法
export function kMeansClustering(
  data: number[][],
  k: number,
  maxIterations: number = 100
): KMeansResult {
  if (data.length === 0 || k <= 0) {
    return {
      centroids: [],
      clusters: [],
      iterations: 0,
      converged: false,
      clusterStats: [],
    };
  }
  
  const n = data.length;
  const dimensions = data[0].length;
  
  // 初始化聚类中心（随机选择k个数据点）
  const centroids: number[][] = [];
  const indices: number[] = [];
  while (indices.length < k) {
    const idx = Math.floor(Math.random() * n);
    if (!indices.includes(idx)) {
      indices.push(idx);
      centroids.push([...data[idx]]);
    }
  }
  
  let clusters: number[] = new Array(n).fill(0);
  let converged = false;
  let iterations = 0;
  
  // 迭代过程
  for (let iter = 0; iter < maxIterations; iter++) {
    iterations++;
    
    // 分配每个点到最近的聚类中心
    const newClusters: number[] = data.map(point => {
      let minDist = Infinity;
      let cluster = 0;
      
      for (let i = 0; i < k; i++) {
        const dist = euclideanDistance(point, centroids[i]);
        if (dist < minDist) {
          minDist = dist;
          cluster = i;
        }
      }
      
      return cluster;
    });
    
    // 检查是否收敛
    if (arraysEqual(clusters, newClusters)) {
      converged = true;
      break;
    }
    
    clusters = newClusters;
    
    // 更新聚类中心
    for (let i = 0; i < k; i++) {
      const clusterPoints = data.filter((_, j) => clusters[j] === i);
      if (clusterPoints.length > 0) {
        centroids[i] = calculateMean(clusterPoints);
      }
    }
  }
  
  // 计算聚类统计信息
  const clusterStats = calculateClusterStats(data, clusters, k);
  
  return {
    centroids,
    clusters,
    iterations,
    converged,
    clusterStats,
  };
}

// 计算欧氏距离
function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(
    a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0)
  );
}

// 计算均值向量
function calculateMean(points: number[][]): number[] {
  if (points.length === 0) return [];
  
  const dimensions = points[0].length;
  const mean: number[] = [];
  
  for (let i = 0; i < dimensions; i++) {
    const sum = points.reduce((acc, point) => acc + point[i], 0);
    mean.push(sum / points.length);
  }
  
  return mean;
}

// 检查两个数组是否相等
function arraysEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((val, i) => val === b[i]);
}

// 计算聚类统计信息
function calculateClusterStats(
  data: number[][],
  clusters: number[],
  k: number
): KMeansResult['clusterStats'] {
  const stats: KMeansResult['clusterStats'] = [];
  
  for (let i = 0; i < k; i++) {
    const clusterIndices = clusters
      .map((c, j) => c === i ? j : -1)
      .filter(j => j >= 0);
    
    const clusterPoints = clusterIndices.map(j => data[j]);
    
    // 计算平均评分（假设第一个特征是评分）
    const avgRating = clusterPoints.length > 0
      ? clusterPoints.reduce((sum, p) => sum + p[0] * 5, 0) / clusterPoints.length
      : 0;
    
    // 确定主导情感（基于特征3-5）
    let positiveCount = 0;
    let negativeCount = 0;
    let neutralCount = 0;
    
    clusterPoints.forEach(p => {
      if (p[2] > 0.5) positiveCount++;
      else if (p[3] > 0.5) negativeCount++;
      else neutralCount++;
    });
    
    const dominantSentiment = positiveCount >= negativeCount && positiveCount >= neutralCount
      ? 'positive'
      : negativeCount >= neutralCount ? 'negative' : 'neutral';
    
    stats.push({
      size: clusterIndices.length,
      avgRating,
      dominantSentiment,
    });
  }
  
  return stats;
}

// 计算轮廓系数（评估聚类质量）
export function calculateSilhouetteScore(
  data: number[][],
  clusters: number[]
): number {
  if (data.length === 0 || clusters.length === 0) return 0;
  
  const n = data.length;
  const silhouetteValues: number[] = [];
  
  for (let i = 0; i < n; i++) {
    const clusterI = clusters[i];
    
    // 计算a(i)：同簇平均距离
    const sameClusterPoints = data
      .filter((_, j) => clusters[j] === clusterI && j !== i);
    
    const a = sameClusterPoints.length > 0
      ? sameClusterPoints.reduce((sum, p) => sum + euclideanDistance(data[i], p), 0) / sameClusterPoints.length
      : 0;
    
    // 计算b(i)：最近异簇平均距离
    const otherClusters = [...new Set(clusters)].filter(c => c !== clusterI);
    let minB = Infinity;
    
    for (const c of otherClusters) {
      const otherClusterPoints = data.filter((_, j) => clusters[j] === c);
      if (otherClusterPoints.length > 0) {
        const avgDist = otherClusterPoints.reduce(
          (sum, p) => sum + euclideanDistance(data[i], p), 0
        ) / otherClusterPoints.length;
        minB = Math.min(minB, avgDist);
      }
    }
    
    const b = minB === Infinity ? 0 : minB;
    
    // 计算轮廓值
    const silhouette = b - a === 0 ? 0 : (b - a) / Math.max(a, b);
    silhouetteValues.push(silhouette);
  }
  
  // 返回平均轮廓系数
  return silhouetteValues.reduce((sum, s) => sum + s, 0) / silhouetteValues.length;
}

// PCA降维算法
export function performPCA(data: number[][], components: number = 2): PCAResult {
  if (data.length === 0) return {
    reducedData: [],
    components: [],
    explainedVariance: [],
    cumulativeVariance: [],
  };
  
  const n = data.length;
  const dimensions = data[0].length;
  
  // 计算均值并中心化数据
  const means: number[] = [];
  for (let j = 0; j < dimensions; j++) {
    means.push(data.reduce((sum, row) => sum + row[j], 0) / n);
  }
  
  const centeredData = data.map(row =>
    row.map((val, j) => val - means[j])
  );
  
  // 计算协方差矩阵
  const covarianceMatrix: number[][] = [];
  for (let i = 0; i < dimensions; i++) {
    covarianceMatrix.push([]);
    for (let j = 0; j < dimensions; j++) {
      const cov = centeredData.reduce((sum, row) => sum + row[i] * row[j], 0) / (n - 1);
      covarianceMatrix[i].push(cov);
    }
  }
  
  // 使用简化方法计算特征值和特征向量
  // 这里使用迭代方法近似求解
  const eigenResult = calculateEigenValues(covarianceMatrix, components);
  
  // 选择主成分
  const sortedEigen = eigenResult.eigenvalues
    .map((val, i) => ({ val, vector: eigenResult.eigenvectors[i] }))
    .sort((a, b) => b.val - a.val)
    .slice(0, components);
  
  // 投影到主成分空间
  const reducedData = centeredData.map(row =>
    sortedEigen.map(eigen =>
      row.reduce((sum, val, j) => sum + val * eigen.vector[j], 0)
    )
  );
  
  // 计算解释方差比例
  const totalVariance = eigenResult.eigenvalues.reduce((sum, val) => sum + val, 0);
  const explainedVariance = sortedEigen.map(eigen => eigen.val / totalVariance);
  
  // 计算累积方差
  const cumulativeVariance: number[] = [];
  let cumulative = 0;
  for (const variance of explainedVariance) {
    cumulative += variance;
    cumulativeVariance.push(cumulative);
  }
  
  return {
    reducedData,
    components: sortedEigen.map(eigen => eigen.vector),
    explainedVariance,
    cumulativeVariance,
  };
}

// 简化的特征值计算（使用幂迭代法）
function calculateEigenValues(
  matrix: number[][],
  numEigenvalues: number
): { eigenvalues: number[]; eigenvectors: number[][] } {
  const n = matrix.length;
  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = [];
  
  let currentMatrix = matrix;
  
  for (let k = 0; k < numEigenvalues; k++) {
    // 初始化随机向量
    let vector: number[] = [];
    for (let i = 0; i < n; i++) {
      vector.push(Math.random());
    }
    
    // 幂迭代
    for (let iter = 0; iter < 100; iter++) {
      // 计算矩阵乘向量
      const newVector: number[] = [];
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          sum += currentMatrix[i][j] * vector[j];
        }
        newVector.push(sum);
      }
      
      // 归一化
      const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
      if (norm > 0) {
        vector = newVector.map(val => val / norm);
      }
    }
    
    // 计算特征值（Rayleigh商）
    let eigenvalue = 0;
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += currentMatrix[i][j] * vector[j];
      }
      eigenvalue += vector[i] * sum;
    }
    
    eigenvalues.push(eigenvalue);
    eigenvectors.push(vector);
    
    // 移除已找到的特征值对应的成分
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        currentMatrix[i][j] -= eigenvalue * vector[i] * vector[j];
      }
    }
  }
  
  return { eigenvalues, eigenvectors };
}
```

---

### 文件4: src/lib/dl-model.ts

创建 `src/lib/dl-model.ts` 文件，复制以下完整内容：

```typescript
// ==================== src/lib/dl-model.ts ====================
// 深度学习模型 - 神经网络情感分类器

import * as tf from '@tensorflow/tfjs';
import { NeuralNetworkConfig, TrainingHistory, PredictionResult, DLResult } from '@/types';

// 神经网络分类器类
export class NeuralNetworkClassifier {
  private model: tf.LayersModel | null = null;
  private config: NeuralNetworkConfig;
  
  constructor(config: NeuralNetworkConfig) {
    this.config = config;
  }
  
  // 构建模型
  buildModel(): void {
    this.model = tf.sequential();
    
    // 输入层
    this.model.add(tf.layers.dense({
      units: this.config.hiddenLayers[0],
      inputShape: [this.config.inputSize],
      activation: 'relu',
    }));
    
    // 添加Dropout层
    this.model.add(tf.layers.dropout({
      rate: this.config.dropoutRate,
    }));
    
    // 隐藏层
    for (let i = 1; i < this.config.hiddenLayers.length; i++) {
      this.model.add(tf.layers.dense({
        units: this.config.hiddenLayers[i],
        activation: 'relu',
      }));
      
      this.model.add(tf.layers.dropout({
        rate: this.config.dropoutRate,
      }));
    }
    
    // 输出层
    this.model.add(tf.layers.dense({
      units: this.config.outputSize,
      activation: 'softmax',
    }));
    
    // 编译模型
    this.model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  }
  
  // 训练模型
  async train(
    XTrain: number[][],
    yTrain: number[][],
    onProgress?: (epoch: number, logs: tf.Logs) => void
  ): Promise<TrainingHistory[]> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }
    
    const history: TrainingHistory[] = [];
    
    // 转换数据为Tensor
    const xTensor = tf.tensor2d(XTrain);
    const yTensor = tf.tensor2d(yTrain);
    
    // 训练
    await this.model.fit(xTensor, yTensor, {
      epochs: this.config.epochs,
      batchSize: this.config.batchSize,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (logs) {
            history.push({
              epoch: epoch + 1,
              loss: logs.loss || 0,
              accuracy: logs.acc || 0,
              valLoss: logs.val_loss,
              valAccuracy: logs.val_acc,
            });
            
            if (onProgress) {
              onProgress(epoch + 1, logs);
            }
          }
        },
      },
    });
    
    // 清理Tensor
    xTensor.dispose();
    yTensor.dispose();
    
    return history;
  }
  
  // 预测
  async predict(XTest: number[][]): Promise<number[]> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }
    
    const xTensor = tf.tensor2d(XTest);
    const predictions = this.model.predict(xTensor) as tf.Tensor;
    const data = predictions.dataSync();
    
    xTensor.dispose();
    predictions.dispose();
    
    // 转换为类别索引
    const result: number[] = [];
    for (let i = 0; i < XTest.length; i++) {
      let maxIdx = 0;
      let maxVal = 0;
      for (let j = 0; j < this.config.outputSize; j++) {
        if (data[i * this.config.outputSize + j] > maxVal) {
          maxVal = data[i * this.config.outputSize + j];
          maxIdx = j;
        }
      }
      result.push(maxIdx);
    }
    
    return result;
  }
  
  // 评估模型
  async evaluate(
    XTest: number[][],
    yTest: number[][]
  ): Promise<PredictionResult> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel() first.');
    }
    
    const predictions = await this.predict(XTest);
    const probabilities = await this.getProbabilities(XTest);
    
    // 计算准确率
    const yTrue = yTest.map(y => y.indexOf(1));
    const correctCount = predictions.reduce(
      (count, pred, i) => count + (pred === yTrue[i] ? 1 : 0),
      0
    );
    const accuracy = correctCount / predictions.length;
    
    // 计算混淆矩阵
    const confusionMatrix = this.calculateConfusionMatrix(predictions, yTrue);
    
    // 计算精确率、召回率、F1分数
    const metrics = this.calculateMetrics(confusionMatrix);
    
    return {
      predictions,
      probabilities,
      accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      f1Score: metrics.f1Score,
      confusionMatrix,
    };
  }
  
  // 获取概率分布
  private async getProbabilities(XTest: number[][]): Promise<number[][]> {
    if (!this.model) {
      throw new Error('Model not built.');
    }
    
    const xTensor = tf.tensor2d(XTest);
    const predictions = this.model.predict(xTensor) as tf.Tensor;
    const data = predictions.dataSync();
    
    xTensor.dispose();
    predictions.dispose();
    
    const result: number[][] = [];
    for (let i = 0; i < XTest.length; i++) {
      const probs: number[] = [];
      for (let j = 0; j < this.config.outputSize; j++) {
        probs.push(data[i * this.config.outputSize + j]);
      }
      result.push(probs);
    }
    
    return result;
  }
  
  // 计算混淆矩阵
  private calculateConfusionMatrix(
    predictions: number[],
    yTrue: number[]
  ): number[][] {
    const numClasses = this.config.outputSize;
    const matrix: number[][] = Array(numClasses)
      .fill(null)
      .map(() => Array(numClasses).fill(0));
    
    for (let i = 0; i < predictions.length; i++) {
      matrix[yTrue[i]][predictions[i]]++;
    }
    
    return matrix;
  }
  
  // 计算评估指标
  private calculateMetrics(confusionMatrix: number[][]): {
    precision: number;
    recall: number;
    f1Score: number;
  } {
    const numClasses = this.config.outputSize;
    
    // 计算每个类别的精确率和召回率
    let totalPrecision = 0;
    let totalRecall = 0;
    let totalF1 = 0;
    
    for (let c = 0; c < numClasses; c++) {
      // 精确率：TP / (TP + FP)
      const tp = confusionMatrix[c][c];
      const fp = confusionMatrix.reduce(
        (sum, row, i) => sum + (i !== c ? row[c] : 0),
        0
      );
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      
      // 召回率：TP / (TP + FN)
      const fn = confusionMatrix[c].reduce(
        (sum, val, i) => sum + (i !== c ? val : 0),
        0
      );
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      
      // F1分数
      const f1 = precision + recall > 0
        ? 2 * precision * recall / (precision + recall)
        : 0;
      
      totalPrecision += precision;
      totalRecall += recall;
      totalF1 += f1;
    }
    
    return {
      precision: totalPrecision / numClasses,
      recall: totalRecall / numClasses,
      f1Score: totalF1 / numClasses,
    };
  }
  
  // 获取模型
  getModel(): tf.LayersModel | null {
    return this.model;
  }
}

// 准备训练数据
export function prepareTrainingData(
  features: number[][],
  labels: ('positive' | 'negative' | 'neutral')[],
  trainRatio: number = 0.8
): {
  XTrain: number[][];
  yTrain: number[][];
  XTest: number[][];
  yTest: number[][];
} {
  const n = features.length;
  const trainSize = Math.floor(n * trainRatio);
  
  // 转换标签为向量
  const labelVectors = labels.map(label => {
    if (label === 'positive') return [1, 0, 0];
    if (label === 'negative') return [0, 1, 0];
    return [0, 0, 1];
  });
  
  // 随机划分训练集和测试集
  const indices = Array.from({ length: n }, (_, i) => i);
  shuffleArray(indices);
  
  const trainIndices = indices.slice(0, trainSize);
  const testIndices = indices.slice(trainSize);
  
  const XTrain = trainIndices.map(i => features[i]);
  const yTrain = trainIndices.map(i => labelVectors[i]);
  const XTest = testIndices.map(i => features[i]);
  const yTest = testIndices.map(i => labelVectors[i]);
  
  return { XTrain, yTrain, XTest, yTest };
}

// 随机打乱数组
function shuffleArray(array: number[]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

// 训练和评估完整流程
export async function trainAndEvaluate(
  features: number[][],
  labels: ('positive' | 'negative' | 'neutral')[],
  config: NeuralNetworkConfig,
  onProgress?: (epoch: number, logs: tf.Logs) => void
): Promise<DLResult> {
  // 准备数据
  const { XTrain, yTrain, XTest, yTest } = prepareTrainingData(features, labels);
  
  // 创建并训练模型
  const classifier = new NeuralNetworkClassifier(config);
  classifier.buildModel();
  
  const history = await classifier.train(XTrain, yTrain, onProgress);
  
  // 评估模型
  const prediction = await classifier.evaluate(XTest, yTest);
  
  return {
    model: classifier.getModel(),
    history,
    prediction,
  };
}
```

---

## 下一步

复制完这4个文件后，你还需要：
1. 替换 `src/app/page.tsx`（主页面）
2. 运行项目

由于 page.tsx 文件较长（约1500行），请在项目中查看完整内容。