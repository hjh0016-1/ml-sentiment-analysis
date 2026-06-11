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