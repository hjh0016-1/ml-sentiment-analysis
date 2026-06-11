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
 * 从GitHub API获取真实数据
 */
export async function fetchGitHubReviews(count: number = 50): Promise<ReviewData[]> {
  try {
    const response = await fetch(`/api/github-data?count=${count}`);
    const result = await response.json();
    
    if (result.success && result.data) {
      return result.data;
    }
    
    // 如果API获取失败，返回模拟数据
    console.warn('GitHub API failed, using mock data');
    return generateMockReviews(count);
  } catch (error) {
    console.error('Error fetching GitHub data:', error);
    return generateMockReviews(count);
  }
}

/**
 * 生成模拟评论数据（备用）
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