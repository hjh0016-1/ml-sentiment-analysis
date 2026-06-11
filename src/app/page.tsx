'use client';

import { useState, useCallback, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
} from 'recharts';
import type {
  ReviewData,
  DataStats,
  FeatureVector,
  KMeansResult,
  PCAResult,
  TrainingResult,
  PredictionResult,
} from '@/types';
import {
  generateMockReviews,
  calculateDataStats,
  preprocessData,
  extractFeatures,
  normalizeFeatures,
} from '@/lib/data-processing';
import { kMeansClustering, performPCA, calculateSilhouetteScore } from '@/lib/ml-algorithms';
import {
  NeuralNetworkClassifier,
  createDefaultNNConfig,
  splitTrainTest,
} from '@/lib/dl-model';
import '@tensorflow/tfjs';

// 颜色配置
const COLORS = {
  positive: '#10b981',
  negative: '#ef4444',
  neutral: '#f59e0b',
  cluster: ['#1e40af', '#f97316', '#10b981', '#8b5cf6', '#ec4899'],
};

export default function DataMiningApp() {
  // 状态管理
  const [activeTab, setActiveTab] = useState('crawler');
  const [rawData, setRawData] = useState<ReviewData[]>([]);
  const [processedData, setProcessedData] = useState<ReviewData[]>([]);
  const [featureVectors, setFeatureVectors] = useState<FeatureVector[]>([]);
  const [stats, setStats] = useState<DataStats | null>(null);
  const [crawlerStatus, setCrawlerStatus] = useState({
    status: 'idle',
    progress: 0,
    collected: 0,
  });

  // 预处理配置
  const [preprocessConfig, setPreprocessConfig] = useState({
    removeDuplicates: true,
    removeMissing: true,
    removeOutliers: false,
    normalizeText: true,
    keywordExtraction: true,
  });

  // ML配置
  const [kmeansConfig, setKmeansConfig] = useState({
    k: 3,
    maxIterations: 100,
    tolerance: 0.001,
  });
  const [kmeansResult, setKmeansResult] = useState<KMeansResult | null>(null);
  const [pcaResult, setPcaResult] = useState<PCAResult | null>(null);

  // DL配置
  const [nnConfig, setNNConfig] = useState({
    epochs: 50,
    learningRate: 0.001,
    batchSize: 32,
  });
  const [dlResult, setDLResult] = useState<{
    training: TrainingResult | null;
    prediction: PredictionResult | null;
  }>({ training: null, prediction: null });
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  // ==================== 数据爬取模块 ====================
  const handleCrawlData = useCallback(() => {
    setCrawlerStatus({ status: 'running', progress: 0, collected: 0 });

    // 模拟爬虫进度
    const total = 200;
    let current = 0;

    const interval = setInterval(() => {
      current += 20;
      setCrawlerStatus({
        status: 'running',
        progress: Math.min((current / total) * 100, 100),
        collected: current,
      });

      if (current >= total) {
        clearInterval(interval);
        const data = generateMockReviews(total);
        setRawData(data);
        setStats(calculateDataStats(data));
        setCrawlerStatus({ status: 'completed', progress: 100, collected: total });
      }
    }, 200);
  }, []);

  // ==================== 数据预处理模块 ====================
  const handlePreprocess = useCallback(() => {
    if (rawData.length === 0) return;

    const processed = preprocessData(rawData, preprocessConfig);
    setProcessedData(processed);

    const features = extractFeatures(processed);
    const normalized = normalizeFeatures(features);
    setFeatureVectors(normalized);
    setStats(calculateDataStats(processed));
  }, [rawData, preprocessConfig]);

  // ==================== 机器学习模块 ====================
  const handleKMeans = useCallback(() => {
    if (featureVectors.length === 0) return;

    try {
      const result = kMeansClustering(featureVectors, {
        k: kmeansConfig.k,
        maxIterations: kmeansConfig.maxIterations,
        tolerance: kmeansConfig.tolerance,
        distanceMetric: 'euclidean',
      });
      setKmeansResult(result);
    } catch (error) {
      console.error('K-means 聚类失败:', error);
    }
  }, [featureVectors, kmeansConfig]);

  const handlePCA = useCallback(() => {
    if (featureVectors.length === 0) return;

    try {
      const result = performPCA(featureVectors, {
        targetDimensions: 2,
      });
      setPcaResult(result);
    } catch (error) {
      console.error('PCA 降维失败:', error);
    }
  }, [featureVectors]);

  // ==================== 深度学习模块 ====================
  const handleTrainModel = useCallback(async () => {
    if (featureVectors.length === 0) return;

    setIsTraining(true);
    setTrainingProgress(0);

    try {
      // 划分数据集
      const { train, test } = splitTrainTest(featureVectors, 0.2);

      // 创建模型
      const config = createDefaultNNConfig();
      config.epochs = nnConfig.epochs;
      config.learningRate = nnConfig.learningRate;
      config.batchSize = nnConfig.batchSize;

      const classifier = new NeuralNetworkClassifier(config);
      classifier.buildModel(featureVectors[0].features.length);

      // 训练模型（模拟进度）
      const progressInterval = setInterval(() => {
        setTrainingProgress((prev) => Math.min(prev + 2, 95));
      }, 100);

      const trainingResult = await classifier.train(train);
      clearInterval(progressInterval);
      setTrainingProgress(100);

      // 预测
      const predictionResult = await classifier.predict(test);

      setDLResult({
        training: trainingResult,
        prediction: predictionResult,
      });

      classifier.dispose();
    } catch (error) {
      console.error('模型训练失败:', error);
    } finally {
      setIsTraining(false);
    }
  }, [featureVectors, nnConfig]);

  // ==================== 可视化数据准备 ====================
  const sentimentPieData = stats
    ? [
        { name: '正向', value: stats.positive, color: COLORS.positive },
        { name: '负向', value: stats.negative, color: COLORS.negative },
        { name: '中性', value: stats.neutral, color: COLORS.neutral },
      ]
    : [];

  const ratingDistribution =
    rawData.length > 0
      ? [1, 2, 3, 4, 5].map((rating) => ({
          rating: `${rating}星`,
          count: rawData.filter((r) => r.rating === rating).length,
        }))
      : [];

  const clusterScatterData =
    pcaResult && kmeansResult
      ? pcaResult.reducedData.map((point, i) => ({
          x: point[0],
          y: point[1],
          cluster: kmeansResult.clusters[i],
        }))
      : [];

  const trainingChartData =
    dlResult.training?.history.loss.map((loss, i) => ({
      epoch: i + 1,
      loss,
      accuracy: dlResult.training?.history.accuracy[i] || 0,
    })) || [];

  const confusionMatrixData =
    dlResult.prediction?.confusionMatrix.flatMap((row, i) =>
      row.map((val, j) => ({
        actual: ['负向', '中性', '正向'][i],
        predicted: ['负向', '中性', '正向'][j],
        value: val,
      }))
    ) || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-900">
      {/* 顶部标题 */}
      <div className="border-b bg-white/80 dark:bg-slate-800/80 backdrop-blur">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
                数据挖掘与机器学习实践平台
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                电商产品评论情感分析 - 期末大作业项目
              </p>
            </div>
            <Badge variant="outline" className="px-4 py-2">
              数据科学实验平台
            </Badge>
          </div>
        </div>
      </div>

      {/* 主内容区 */}
      <div className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 gap-2 bg-white/60 dark:bg-slate-800/60 p-1">
            <TabsTrigger value="crawler" className="data-[state=active]:bg-blue-600">
              数据采集
            </TabsTrigger>
            <TabsTrigger value="preprocess" className="data-[state=active]:bg-blue-600">
              数据预处理
            </TabsTrigger>
            <TabsTrigger value="ml" className="data-[state=active]:bg-blue-600">
              传统ML
            </TabsTrigger>
            <TabsTrigger value="dl" className="data-[state=active]:bg-blue-600">
              深度学习
            </TabsTrigger>
            <TabsTrigger value="comparison" className="data-[state=active]:bg-blue-600">
              结果对比
            </TabsTrigger>
          </TabsList>

          {/* ==================== 数据采集模块 ==================== */}
          <TabsContent value="crawler" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="text-2xl">🔍</span>
                  数据爬取模块
                </CardTitle>
                <CardDescription>
                  模拟网络爬虫采集电商产品评论数据（支持公开数据集导入）
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4">
                  <Button
                    onClick={handleCrawlData}
                    disabled={crawlerStatus.status === 'running'}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    {crawlerStatus.status === 'running'
                      ? '正在采集...'
                      : '开始采集数据'}
                  </Button>
                  <Badge
                    variant={
                      crawlerStatus.status === 'completed'
                        ? 'default'
                        : 'outline'
                    }
                    className={
                      crawlerStatus.status === 'completed'
                        ? 'bg-green-600'
                        : ''
                    }
                  >
                    {crawlerStatus.status === 'idle'
                      ? '等待开始'
                      : crawlerStatus.status === 'running'
                        ? '采集进行中'
                        : '采集完成'}
                  </Badge>
                </div>

                {crawlerStatus.status !== 'idle' && (
                  <div className="space-y-2">
                    <Label>采集进度</Label>
                    <Progress value={crawlerStatus.progress} className="h-3" />
                    <p className="text-sm text-slate-600">
                      已采集 {crawlerStatus.collected} 条数据
                    </p>
                  </div>
                )}

                {stats && (
                  <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">数据概览</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center p-4 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-3xl font-bold text-blue-600">
                            {stats.total}
                          </p>
                          <p className="text-sm text-slate-600">总评论数</p>
                        </div>
                        <div className="text-center p-4 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-3xl font-bold text-green-600">
                            {stats.avgRating}
                          </p>
                          <p className="text-sm text-slate-600">平均评分</p>
                        </div>
                        <div className="text-center p-4 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-3xl font-bold text-orange-600">
                            {stats.avgLength}
                          </p>
                          <p className="text-sm text-slate-600">平均长度</p>
                        </div>
                        <div className="text-center p-4 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-3xl font-bold text-red-600">
                            {stats.outliers}
                          </p>
                          <p className="text-sm text-slate-600">异常数据</p>
                        </div>
                      </div>

                      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="h-64">
                          <p className="text-sm font-medium mb-2 text-center">
                            情感分布
                          </p>
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={sentimentPieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                                label={({ name, percent }) =>
                                  `${name} ${(percent * 100).toFixed(0)}%`
                                }
                              >
                                {sentimentPieData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                              <Tooltip />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>

                        <div className="h-64">
                          <p className="text-sm font-medium mb-2 text-center">
                            评分分布
                          </p>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={ratingDistribution}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="rating" />
                              <YAxis />
                              <Tooltip />
                              <Bar dataKey="count" fill="#1e40af" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {rawData.length > 0 && (
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">数据样本预览</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ScrollArea className="h-64">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>产品</TableHead>
                              <TableHead>评论内容</TableHead>
                              <TableHead>评分</TableHead>
                              <TableHead>情感</TableHead>
                              <TableHead>关键词</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {rawData.slice(0, 50).map((review) => (
                              <TableRow key={review.id}>
                                <TableCell className="font-medium">
                                  {review.productName}
                                </TableCell>
                                <TableCell className="max-w-xs truncate">
                                  {review.content}
                                </TableCell>
                                <TableCell>
                                  <Badge
                                    variant="outline"
                                    className={
                                      review.rating >= 4
                                        ? 'border-green-600 text-green-600'
                                        : review.rating <= 2
                                          ? 'border-red-600 text-red-600'
                                          : 'border-orange-600 text-orange-600'
                                    }
                                  >
                                    {review.rating}星
                                  </Badge>
                                </TableCell>
                                <TableCell>
                                  <Badge
                                    className={
                                      review.sentiment === 'positive'
                                        ? 'bg-green-600'
                                        : review.sentiment === 'negative'
                                          ? 'bg-red-600'
                                          : 'bg-orange-600'
                                    }
                                  >
                                    {review.sentiment === 'positive'
                                      ? '正向'
                                      : review.sentiment === 'negative'
                                        ? '负向'
                                        : '中性'}
                                  </Badge>
                                </TableCell>
                                <TableCell className="text-xs">
                                  {review.keywords.join(', ')}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </ScrollArea>
                    </CardContent>
                  </Card>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* ==================== 数据预处理模块 ==================== */}
          <TabsContent value="preprocess" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="text-2xl">🧹</span>
                  数据预处理模块
                </CardTitle>
                <CardDescription>
                  数据清洗、转换、特征提取与探索性分析
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* 预处理配置 */}
                <Card className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800/40 dark:to-slate-700/40">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">预处理配置</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={preprocessConfig.removeDuplicates}
                          onCheckedChange={(checked) =>
                            setPreprocessConfig({
                              ...preprocessConfig,
                              removeDuplicates: checked,
                            })
                          }
                        />
                        <Label>移除重复</Label>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={preprocessConfig.removeMissing}
                          onCheckedChange={(checked) =>
                            setPreprocessConfig({
                              ...preprocessConfig,
                              removeMissing: checked,
                            })
                          }
                        />
                        <Label>移除缺失值</Label>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={preprocessConfig.removeOutliers}
                          onCheckedChange={(checked) =>
                            setPreprocessConfig({
                              ...preprocessConfig,
                              removeOutliers: checked,
                            })
                          }
                        />
                        <Label>移除异常值</Label>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={preprocessConfig.normalizeText}
                          onCheckedChange={(checked) =>
                            setPreprocessConfig({
                              ...preprocessConfig,
                              normalizeText: checked,
                            })
                          }
                        />
                        <Label>文本标准化</Label>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Button
                  onClick={handlePreprocess}
                  disabled={rawData.length === 0}
                  className="bg-green-600 hover:bg-green-700"
                >
                  执行预处理
                </Button>

                {processedData.length > 0 && (
                  <div className="space-y-4">
                    {/* 处理结果统计 */}
                    <div className="grid grid-cols-3 gap-4">
                      <Card className="bg-blue-50/50 dark:bg-blue-900/20">
                        <CardContent className="pt-4">
                          <p className="text-3xl font-bold text-blue-600">
                            {rawData.length}
                          </p>
                          <p className="text-sm">原始数据</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-green-50/50 dark:bg-green-900/20">
                        <CardContent className="pt-4">
                          <p className="text-3xl font-bold text-green-600">
                            {processedData.length}
                          </p>
                          <p className="text-sm">处理后数据</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-red-50/50 dark:bg-red-900/20">
                        <CardContent className="pt-4">
                          <p className="text-3xl font-bold text-red-600">
                            {rawData.length - processedData.length}
                          </p>
                          <p className="text-sm">移除数据</p>
                        </CardContent>
                      </Card>
                    </div>

                    {/* 特征向量信息 */}
                    {featureVectors.length > 0 && (
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-lg">
                            特征向量提取结果
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                              <p className="text-sm font-medium">特征维度</p>
                              <p className="text-2xl font-bold text-blue-600">
                                {featureVectors[0].features.length}
                              </p>
                            </div>
                            <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                              <p className="text-sm font-medium">样本数量</p>
                              <p className="text-2xl font-bold text-green-600">
                                {featureVectors.length}
                              </p>
                            </div>
                            <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                              <p className="text-sm font-medium">正向样本</p>
                              <p className="text-2xl font-bold text-green-600">
                                {
                                  featureVectors.filter((f) => f.label === 2)
                                    .length
                                }
                              </p>
                            </div>
                            <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                              <p className="text-sm font-medium">负向样本</p>
                              <p className="text-2xl font-bold text-red-600">
                                {
                                  featureVectors.filter((f) => f.label === 0)
                                    .length
                                }
                              </p>
                            </div>
                          </div>

                          <Separator className="my-4" />

                          <p className="text-sm text-slate-600 mb-2">
                            特征向量示例（前3个样本）：
                          </p>
                          <ScrollArea className="h-48">
                            <div className="space-y-2">
                              {featureVectors.slice(0, 3).map((vector, i) => (
                                <div
                                  key={i}
                                  className="p-2 bg-white/60 dark:bg-slate-700/60 rounded"
                                >
                                  <p className="text-xs font-medium mb-1">
                                    样本 {i + 1}（标签: {vector.label === 2 ? '正向' : vector.label === 1 ? '中性' : '负向'}）
                                  </p>
                                  <p className="text-xs text-slate-600 dark:text-slate-400">
                                    特征: [{vector.features.slice(0, 5).map(f => f.toFixed(2)).join(', ')}...]
                                  </p>
                                </div>
                              ))}
                            </div>
                          </ScrollArea>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* ==================== 传统机器学习模块 ==================== */}
          <TabsContent value="ml" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="text-2xl">📊</span>
                  传统机器学习模块
                </CardTitle>
                <CardDescription>
                  K-means聚类与PCA降维分析
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* K-means配置 */}
                <Card className="bg-gradient-to-r from-blue-50/50 to-indigo-50/50 dark:from-blue-900/20 dark:to-indigo-900/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">K-means 聚类</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>聚类数量 (K)</Label>
                      <div className="flex items-center gap-4">
                        <Slider
                          value={[kmeansConfig.k]}
                          onValueChange={(value) =>
                            setKmeansConfig({ ...kmeansConfig, k: value[0] })
                          }
                          min={2}
                          max={5}
                          step={1}
                          className="w-64"
                        />
                        <Badge variant="outline" className="px-3">
                          K = {kmeansConfig.k}
                        </Badge>
                      </div>
                    </div>

                    <Button
                      onClick={handleKMeans}
                      disabled={featureVectors.length === 0}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      执行K-means聚类
                    </Button>

                    {kmeansResult && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-3 gap-4">
                          <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg text-center">
                            <p className="text-2xl font-bold text-blue-600">
                              {kmeansResult.iterations}
                            </p>
                            <p className="text-sm">迭代次数</p>
                          </div>
                          <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg text-center">
                            <p className="text-2xl font-bold text-green-600">
                              {kmeansResult.converged ? '是' : '否'}
                            </p>
                            <p className="text-sm">是否收敛</p>
                          </div>
                          <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg text-center">
                            <p className="text-2xl font-bold text-orange-600">
                              {calculateSilhouetteScore(
                                featureVectors,
                                kmeansResult.clusters
                              ).toFixed(3)}
                            </p>
                            <p className="text-sm">轮廓系数</p>
                          </div>
                        </div>

                        {/* 聚类统计 */}
                        <Card>
                          <CardHeader className="pb-2">
                            <CardTitle className="text-base">
                              聚类统计详情
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                              {kmeansResult.clusterStats.map((stat) => (
                                <div
                                  key={stat.clusterId}
                                  className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg"
                                  style={{
                                    borderColor: COLORS.cluster[stat.clusterId],
                                    borderWidth: '2px',
                                  }}
                                >
                                  <p className="text-sm font-medium mb-2">
                                    聚类 {stat.clusterId}
                                  </p>
                                  <div className="space-y-1">
                                    <p className="text-xs">
                                      样本数: {stat.size}
                                    </p>
                                    <div className="flex gap-1">
                                      <Badge
                                        className="bg-green-600 text-xs"
                                        variant="outline"
                                      >
                                        正{stat.sentimentDistribution.positive}
                                      </Badge>
                                      <Badge
                                        className="bg-orange-600 text-xs"
                                        variant="outline"
                                      >
                                        中{stat.sentimentDistribution.neutral}
                                      </Badge>
                                      <Badge
                                        className="bg-red-600 text-xs"
                                        variant="outline"
                                      >
                                        负{stat.sentimentDistribution.negative}
                                      </Badge>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* PCA配置 */}
                <Card className="bg-gradient-to-r from-purple-50/50 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">PCA 降维</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Button
                      onClick={handlePCA}
                      disabled={featureVectors.length === 0}
                      className="bg-purple-600 hover:bg-purple-700"
                    >
                      执行PCA降维
                    </Button>

                    {pcaResult && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                            <p className="text-sm font-medium mb-2">
                              解释方差比例
                            </p>
                            <div className="space-y-1">
                              {pcaResult.explainedVariance.map((variance, i) => (
                                <div key={i} className="flex items-center gap-2">
                                  <Badge variant="outline">PC{i + 1}</Badge>
                                  <Progress
                                    value={variance * 100}
                                    className="h-2 flex-1"
                                  />
                                  <span className="text-xs">
                                    {(variance * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                          <div className="p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                            <p className="text-sm font-medium mb-2">
                              累积方差
                            </p>
                            <p className="text-2xl font-bold text-purple-600">
                              {(
                                pcaResult.cumulativeVariance[
                                  pcaResult.cumulativeVariance.length - 1
                                ] * 100
                              ).toFixed(1)}
                              %
                            </p>
                            <p className="text-xs text-slate-600">
                              前2个主成分保留的原始信息量
                            </p>
                          </div>
                        </div>

                        {/* PCA可视化 */}
                        {clusterScatterData.length > 0 && (
                          <div className="h-64">
                            <p className="text-sm font-medium mb-2 text-center">
                              PCA降维可视化（结合K-means聚类结果）
                            </p>
                            <ResponsiveContainer width="100%" height="100%">
                              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid />
                                <XAxis type="number" dataKey="x" name="PC1" />
                                <YAxis type="number" dataKey="y" name="PC2" />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                <Legend />
                                {[0, 1, 2, 3, 4].slice(0, kmeansConfig.k).map((clusterId) => (
                                  <Scatter
                                    key={clusterId}
                                    name={`聚类 ${clusterId}`}
                                    data={clusterScatterData.filter(d => d.cluster === clusterId)}
                                    fill={COLORS.cluster[clusterId]}
                                  />
                                ))}
                              </ScatterChart>
                            </ResponsiveContainer>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* 算法说明 */}
                <Card className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800/40 dark:to-slate-700/40">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">算法原理说明</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="p-3 bg-blue-50/30 dark:bg-blue-900/10 rounded-lg">
                      <p className="text-sm font-medium text-blue-600 mb-1">
                        K-means聚类算法
                      </p>
                      <p className="text-xs text-slate-600 dark:text-slate-400">
                        迭代算法，通过计算样本与聚类中心的距离，将数据划分为K个簇。
                        每次迭代更新聚类中心直到收敛。轮廓系数用于评估聚类质量。
                      </p>
                    </div>
                    <div className="p-3 bg-purple-50/30 dark:bg-purple-900/10 rounded-lg">
                      <p className="text-sm font-medium text-purple-600 mb-1">
                        PCA降维算法
                      </p>
                      <p className="text-xs text-slate-600 dark:text-slate-400">
                        通过计算协方差矩阵的特征值和特征向量，找到数据最大方差的方向（主成分），
                        实现维度降低的同时保留最大信息量。
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ==================== 深度学习模块 ==================== */}
          <TabsContent value="dl" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="text-2xl">🧠</span>
                  深度学习模块
                </CardTitle>
                <CardDescription>
                  神经网络情感分类模型（使用TensorFlow.js）
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* 模型配置 */}
                <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">模型配置</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="space-y-2">
                        <Label>训练轮数 (Epochs)</Label>
                        <Slider
                          value={[nnConfig.epochs]}
                          onValueChange={(value) =>
                            setNNConfig({ ...nnConfig, epochs: value[0] })
                          }
                          min={10}
                          max={100}
                          step={10}
                          className="w-full"
                        />
                        <Badge variant="outline">{nnConfig.epochs} 轮</Badge>
                      </div>
                      <div className="space-y-2">
                        <Label>学习率</Label>
                        <Slider
                          value={[nnConfig.learningRate * 1000]}
                          onValueChange={(value) =>
                            setNNConfig({
                              ...nnConfig,
                              learningRate: value[0] / 1000,
                            })
                          }
                          min={1}
                          max={10}
                          step={1}
                          className="w-full"
                        />
                        <Badge variant="outline">
                          {nnConfig.learningRate.toFixed(3)}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <Label>批量大小</Label>
                        <Slider
                          value={[nnConfig.batchSize]}
                          onValueChange={(value) =>
                            setNNConfig({ ...nnConfig, batchSize: value[0] })
                          }
                          min={16}
                          max={64}
                          step={8}
                          className="w-full"
                        />
                        <Badge variant="outline">{nnConfig.batchSize}</Badge>
                      </div>
                    </div>

                    {/* 网络架构展示 */}
                    <Separator className="my-4" />
                    <div className="space-y-2">
                      <Label>神经网络架构</Label>
                      <div className="p-4 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                        <div className="flex items-center gap-2 justify-center">
                          {[
                            { name: '输入层', units: '15维' },
                            { name: '隐藏层1', units: '64' },
                            { name: 'Dropout', units: '30%' },
                            { name: '隐藏层2', units: '32' },
                            { name: 'Dropout', units: '20%' },
                            { name: '隐藏层3', units: '16' },
                            { name: '输出层', units: '3' },
                          ].map((layer, i) => (
                            <div
                              key={i}
                              className="flex flex-col items-center gap-1"
                            >
                              <div
                                className={`w-16 h-12 rounded-lg flex items-center justify-center text-xs font-medium ${
                                  i === 0
                                    ? 'bg-blue-600 text-white'
                                    : i === 6
                                      ? 'bg-green-600 text-white'
                                      : layer.name === 'Dropout'
                                        ? 'bg-orange-600 text-white'
                                        : 'bg-indigo-600 text-white'
                                }`}
                              >
                                {layer.units}
                              </div>
                              <p className="text-xs text-slate-600">
                                {layer.name}
                              </p>
                              {i < 6 && (
                                <div className="w-8 h-0.5 bg-slate-400" />
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <Button
                      onClick={handleTrainModel}
                      disabled={featureVectors.length === 0 || isTraining}
                      className="bg-indigo-600 hover:bg-indigo-700 w-full"
                    >
                      {isTraining ? '训练进行中...' : '开始训练模型'}
                    </Button>

                    {isTraining && (
                      <div className="space-y-2">
                        <Label>训练进度</Label>
                        <Progress value={trainingProgress} className="h-3" />
                        <p className="text-sm text-center text-slate-600">
                          正在第 {Math.floor(trainingProgress / 2)} / {nnConfig.epochs} 轮训练...
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* 训练结果 */}
                {dlResult.training && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <Card className="bg-green-50/50 dark:bg-green-900/20">
                        <CardContent className="pt-4">
                          <p className="text-3xl font-bold text-green-600">
                            {(dlResult.training.finalAccuracy * 100).toFixed(1)}%
                          </p>
                          <p className="text-sm">最终准确率</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-blue-50/50 dark:bg-blue-900/20">
                        <CardContent className="pt-4">
                          <p className="text-3xl font-bold text-blue-600">
                            {dlResult.training.finalLoss.toFixed(4)}
                          </p>
                          <p className="text-sm">最终损失</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-purple-50/50 dark:bg-purple-900/20">
                        <CardContent className="pt-4">
                          <p className="text-3xl font-bold text-purple-600">
                            {dlResult.training.modelParams}
                          </p>
                          <p className="text-sm">模型参数量</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-orange-50/50 dark:bg-orange-900/20">
                        <CardContent className="pt-4">
                          <p className="text-3xl font-bold text-orange-600">
                            {(dlResult.training.trainingTime / 1000).toFixed(1)}s
                          </p>
                          <p className="text-sm">训练耗时</p>
                        </CardContent>
                      </Card>
                    </div>

                    {/* 训练曲线 */}
                    <div className="h-64">
                      <p className="text-sm font-medium mb-2 text-center">
                        训练过程可视化
                      </p>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trainingChartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" />
                          <Tooltip />
                          <Legend />
                          <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="loss"
                            stroke="#ef4444"
                            name="损失"
                          />
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="accuracy"
                            stroke="#10b981"
                            name="准确率"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* 预测结果 */}
                {dlResult.prediction && (
                  <Card className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">模型评估结果</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-4 gap-4">
                        <div className="text-center p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-2xl font-bold text-green-600">
                            {(dlResult.prediction.accuracy * 100).toFixed(1)}%
                          </p>
                          <p className="text-sm">准确率</p>
                        </div>
                        <div className="text-center p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-2xl font-bold text-blue-600">
                            {(dlResult.prediction.precision * 100).toFixed(1)}%
                          </p>
                          <p className="text-sm">精确率</p>
                        </div>
                        <div className="text-center p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-2xl font-bold text-purple-600">
                            {(dlResult.prediction.recall * 100).toFixed(1)}%
                          </p>
                          <p className="text-sm">召回率</p>
                        </div>
                        <div className="text-center p-3 bg-white/60 dark:bg-slate-700/60 rounded-lg">
                          <p className="text-2xl font-bold text-orange-600">
                            {(dlResult.prediction.f1Score * 100).toFixed(1)}%
                          </p>
                          <p className="text-sm">F1分数</p>
                        </div>
                      </div>

                      {/* 混淆矩阵 */}
                      <div className="h-48">
                        <p className="text-sm font-medium mb-2 text-center">
                          混淆矩阵
                        </p>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={confusionMatrixData}
                            layout="vertical"
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" />
                            <YAxis
                              dataKey="actual"
                              type="category"
                              width={60}
                            />
                            <Tooltip />
                            <Legend />
                            <Bar
                              dataKey="value"
                              fill="#1e40af"
                              name="预测数量"
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* ==================== 结果对比模块 ==================== */}
          <TabsContent value="comparison" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span className="text-2xl">📈</span>
                  结果对比与分析
                </CardTitle>
                <CardDescription>
                  传统机器学习与深度学习方法的性能对比
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* 方法对比表格 */}
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-32">方法</TableHead>
                      <TableHead>主要特点</TableHead>
                      <TableHead>适用场景</TableHead>
                      <TableHead className="text-center">优势</TableHead>
                      <TableHead className="text-center">局限</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow>
                      <TableCell className="font-medium text-blue-600">
                        K-means聚类
                      </TableCell>
                      <TableCell>
                        无监督学习，自动发现数据内在结构
                      </TableCell>
                      <TableCell>
                        用户群体划分、市场细分、异常检测
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge className="bg-green-600">简单高效</Badge>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge className="bg-red-600">需预设K值</Badge>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium text-purple-600">
                        PCA降维
                      </TableCell>
                      <TableCell>
                        线性降维，保留最大方差信息
                      </TableCell>
                      <TableCell>
                        高维数据可视化、特征压缩、降噪
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge className="bg-green-600">
                          信息损失可控
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge className="bg-red-600">
                          仅线性变换
                        </Badge>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium text-indigo-600">
                        神经网络
                      </TableCell>
                      <TableCell>
                        深度学习，自动特征学习与分类
                      </TableCell>
                      <TableCell>
                        复杂模式识别、非线性分类、预测
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge className="bg-green-600">
                          高准确率
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge className="bg-red-600">
                          需大量数据
                        </Badge>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>

                {/* 性能对比 */}
                {dlResult.prediction && kmeansResult && (
                  <Card className="bg-gradient-to-r from-slate-50 to-blue-50 dark:from-slate-800/40 dark:to-blue-900/20">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">
                        本实验性能对比
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-6">
                        {/* ML结果 */}
                        <div className="space-y-2">
                          <p className="text-sm font-medium text-blue-600">
                            传统ML方法
                          </p>
                          <div className="p-4 bg-white/60 dark:bg-slate-700/60 rounded-lg space-y-2">
                            <div className="flex justify-between">
                              <span className="text-sm">聚类轮廓系数:</span>
                              <span className="text-sm font-bold text-orange-600">
                                {calculateSilhouetteScore(
                                  featureVectors,
                                  kmeansResult.clusters
                                ).toFixed(3)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm">PCA解释方差:</span>
                              <span className="text-sm font-bold text-purple-600">
                                {pcaResult
                                  ? `${(
                                    pcaResult.cumulativeVariance[
                                      pcaResult.cumulativeVariance.length - 1
                                    ] * 100
                                  ).toFixed(1)}%`
                                  : 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* DL结果 */}
                        <div className="space-y-2">
                          <p className="text-sm font-medium text-indigo-600">
                            深度学习方法
                          </p>
                          <div className="p-4 bg-white/60 dark:bg-slate-700/60 rounded-lg space-y-2">
                            <div className="flex justify-between">
                              <span className="text-sm">分类准确率:</span>
                              <span className="text-sm font-bold text-green-600">
                                {(
                                  dlResult.prediction.accuracy * 100
                                ).toFixed(1)}
                                %
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-sm">F1分数:</span>
                              <span className="text-sm font-bold text-orange-600">
                                {(
                                  dlResult.prediction.f1Score * 100
                                ).toFixed(1)}
                                %
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <Separator className="my-4" />

                      {/* 结论分析 */}
                      <div className="p-4 bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-lg">
                        <p className="text-sm font-medium mb-2">
                          📋 实验结论
                        </p>
                        <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                          <p>
                            1. 传统ML方法适合探索性分析和数据理解，能够快速发现数据内在结构。
                          </p>
                          <p>
                            2. 深度学习方法在分类任务上表现更优，适合有明确标签的预测任务。
                          </p>
                          <p>
                            3. PCA降维有助于可视化高维数据，便于理解聚类结果的分布特征。
                          </p>
                          <p>
                            4. 建议在实际项目中结合多种方法，先用ML探索数据，再用DL进行精确预测。
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* 技术栈说明 */}
                <Card className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800/40 dark:to-slate-700/40">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">
                      技术实现说明
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="p-3 bg-blue-50/30 dark:bg-blue-900/10 rounded-lg text-center">
                        <p className="text-xs font-medium text-blue-600">
                          Next.js 16
                        </p>
                        <p className="text-xs text-slate-600">前端框架</p>
                      </div>
                      <div className="p-3 bg-indigo-50/30 dark:bg-indigo-900/10 rounded-lg text-center">
                        <p className="text-xs font-medium text-indigo-600">
                          TensorFlow.js
                        </p>
                        <p className="text-xs text-slate-600">深度学习</p>
                      </div>
                      <div className="p-3 bg-purple-50/30 dark:bg-purple-900/10 rounded-lg text-center">
                        <p className="text-xs font-medium text-purple-600">
                          Recharts
                        </p>
                        <p className="text-xs text-slate-600">数据可视化</p>
                      </div>
                      <div className="p-3 bg-green-50/30 dark:bg-green-900/10 rounded-lg text-center">
                        <p className="text-xs font-medium text-green-600">
                          shadcn/ui
                        </p>
                        <p className="text-xs text-slate-600">UI组件库</p>
                      </div>
                    </div>

                    <Separator className="my-4" />

                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      <p className="mb-2">
                        本项目实现了完整的数据挖掘与机器学习流程：
                      </p>
                      <ul className="list-disc list-inside space-y-1">
                        <li>
                          数据采集：模拟爬虫采集电商评论数据
                        </li>
                        <li>
                          数据预处理：清洗、转换、特征提取
                        </li>
                        <li>
                          传统ML：K-means聚类 + PCA降维
                        </li>
                        <li>
                          深度学习：多层神经网络分类
                        </li>
                        <li>
                          结果可视化与对比分析
                        </li>
                      </ul>
                    </div>
                  </CardContent>
                </Card>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* 页脚 */}
      <div className="border-t bg-white/60 dark:bg-slate-800/60 mt-8">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-sm text-slate-600 dark:text-slate-400">
            <p>数据挖掘与机器学习实践平台 - 期末大作业</p>
            <p>技术栈: Next.js + TensorFlow.js + Recharts</p>
          </div>
        </div>
      </div>
    </div>
  );
}