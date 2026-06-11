# 项目交付包 - 数据挖掘与机器学习实践平台

## 📦 项目概述

**项目名称**：电商产品评论情感分析系统  
**项目类型**：期末大作业 - 数据挖掘与机器学习实践  
**技术栈**：Next.js 16 + React 19 + TypeScript + TensorFlow.js + Recharts

---

## 📁 项目文件结构

```
项目根目录/
├── .coze                    # 项目配置文件（构建和运行命令）
├── package.json             # 依赖管理
├── tsconfig.json            # TypeScript配置
├── next.config.ts           # Next.js配置
├── AGENTS.md                # 项目说明文档
├── DESIGN.md                # 设计规范文档
├── README.md                # 项目简介
│
├── src/
│   ├── app/
│   │   ├── layout.tsx       # 应用布局
│   │   ├── page.tsx         # ★ 主页面（所有功能模块）
│   │   ├── globals.css      # 全局样式
│   │   └── robots.ts        # SEO配置
│   │
│   ├── lib/
│   │   ├── data-processing.ts  # ★ 数据处理模块
│   │   ├── ml-algorithms.ts    # ★ 传统ML算法
│   │   ├── dl-model.ts         # ★ 深度学习模型
│   │   └── utils.ts            # 通用工具函数
│   │
│   ├── types/
│   │   └── index.ts         # ★ 类型定义
│   │
│   ├── components/
│   │   └── ui/              # shadcn/ui组件库（50+组件）
│   │
│   ├── hooks/
│   │   └── use-mobile.ts    # 移动端检测
│   │
│   └── server.ts            # 服务端配置
│
└── scripts/
    ├── build.sh             # 构建脚本
    ├── dev.sh               # 开发脚本
    ├── start.sh             # 启动脚本
    └── validate.sh          # 验证脚本
```

**★ 标记的文件为核心业务代码**

---

## 🛠️ 本地部署指南

### 方法一：克隆代码仓库

如果你有 Git 仓库访问权限：

```bash
# 1. 克隆项目
git clone <你的仓库地址>
cd projects

# 2. 安装依赖（必须使用 pnpm）
pnpm install

# 3. 启动开发环境
pnpm dev

# 4. 打开浏览器访问
# http://localhost:5000
```

### 方法二：手动创建项目

如果没有 Git 访问权限，可以按以下步骤手动创建：

#### 步骤1：创建项目目录

```bash
mkdir sentiment-analysis-project
cd sentiment-analysis-project
```

#### 步骤2：初始化 Next.js 项目

```bash
# 使用 Coze CLI 初始化（推荐）
coze init . --template nextjs

# 或使用标准 Next.js 初始化
npx create-next-app@latest . --typescript --tailwind --app
```

#### 步骤3：安装额外依赖

```bash
pnpm add @tensorflow/tfjs recharts mathjs papaparse
```

#### 步骤4：创建核心文件

按下方提供的代码内容，依次创建以下文件：
- `src/types/index.ts` - 类型定义
- `src/lib/data-processing.ts` - 数据处理
- `src/lib/ml-algorithms.ts` - ML算法
- `src/lib/dl-model.ts` - 深度学习
- `src/app/page.tsx` - 主页面

---

## 📋 核心代码文件

### 1. 类型定义 (src/types/index.ts)

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

## 🎯 完整使用指南

### 一、启动项目

```bash
# 开发环境
pnpm dev

# 或生产环境
pnpm build
pnpm start
```

访问地址：`http://localhost:5000`

---

### 二、功能模块使用详解

#### 📍 模块1：数据采集

**功能说明**：模拟网络爬虫采集电商产品评论数据

**操作步骤**：
1. 点击顶部导航栏 **"数据采集"** Tab
2. 点击绿色按钮 **"开始采集数据"**
3. 观察进度条变化（0% → 100%）
4. 数据采集完成后，查看：
   - 数据概览卡片（总数、平均评分、平均长度）
   - 情感分布饼图（正向/负向/中性比例）
   - 评分分布柱状图（1-5星分布）
   - 数据表格预览（前50条记录）

**参数说明**：
- 默认采集200条模拟数据
- 包含5种产品：蓝牙耳机、智能手表、机械键盘、充电宝、降噪耳机
- 情感标签：正向(评分≥4)、负向(评分≤2)、中性(评分3)

---

#### 📍 模块2：数据预处理

**功能说明**：对原始数据进行清洗、转换和特征提取

**操作步骤**：
1. 切换到 **"数据预处理"** Tab
2. 配置预处理选项（开关按钮）：
   - ✅ **移除重复**：删除完全相同的评论
   - ✅ **移除缺失值**：删除评分或内容为空的记录
   - ⬜ **移除异常值**：删除内容长度>500的异常记录
   - ✅ **文本标准化**：转小写、去特殊字符
3. 点击绿色按钮 **"执行预处理"**
4. 查看处理结果：
   - 原始数据 → 处理后数据 → 移除数据统计
   - 特征向量信息（15维特征、样本数量）
   - 特征向量示例（前3个样本）

**特征说明**（15维特征向量）：
```
[评分, 归一化评分, 文本长度, 归一化长度, 
 正向关键词数, 负向关键词数, 中性关键词数,
 正向标签编码, 负向标签编码, 中性标签编码,
 是否验证购买, 时间差(天), 产品编码, 
 关键词数量, 归一化关键词数量]
```

---

#### 📍 模块3：传统机器学习

**功能说明**：K-means聚类和PCA降维分析

##### K-means聚类

**操作步骤**：
1. 切换到 **"传统ML"** Tab
2. 使用滑块调节 **聚类数量K**（2-5）
3. 点击蓝色按钮 **"执行K-means聚类"**
4. 查看聚类结果：
   - **迭代次数**：算法收敛需要的迭代数
   - **是否收敛**：是否达到稳定状态
   - **轮廓系数**：聚类质量评估指标（0-1，越高越好）
   - **聚类统计**：每个簇的样本数和情感分布

**参数说明**：
- K值建议：情感分析建议K=3（对应正向/负向/中性）
- 轮廓系数：>0.5表示聚类质量较好

##### PCA降维

**操作步骤**：
1. 点击紫色按钮 **"执行PCA降维"**
2. 查看降维结果：
   - **解释方差比例**：PC1和PC2保留的信息量
   - **累积方差**：两个主成分总共保留的原始信息量
   - **2D散点图**：结合聚类结果的可视化展示

**原理说明**：
- PCA将15维数据降至2维便于可视化
- 解释方差比例表示每个主成分的重要性
- 累积方差>70%表示降维效果良好

---

#### 📍 模块4：深度学习

**功能说明**：神经网络情感分类模型训练与预测

**操作步骤**：
1. 切换到 **"深度学习"** Tab
2. 调节训练参数（三个滑块）：
   - **训练轮数**：10-100轮（建议50轮）
   - **学习率**：0.001-0.01（建议0.001）
   - **批量大小**：16-64（建议32）
3. 查看网络架构图（可视化展示）
4. 点击紫色按钮 **"开始训练模型"**
5. 观察训练进度：
   - 进度条实时更新
   - 训练轮数显示
6. 训练完成后查看结果：
   - **最终准确率**：模型分类准确度
   - **最终损失**：训练损失值
   - **模型参数量**：网络参数总数
   - **训练耗时**：总训练时间
   - **训练曲线**：损失和准确率变化图
   - **评估指标**：准确率、精确率、召回率、F1分数
   - **混淆矩阵**：预测结果分布

**网络架构**：
```
输入层(15维) → Dense(64, relu) → Dropout(0.3)
→ Dense(32, relu) → Dropout(0.2) → Dense(16, relu)
→ 输出层(3, softmax)
```

**参数建议**：
- 快速测试：10轮 + 学习率0.01
- 正常训练：50轮 + 学习率0.001
- 高精度：100轮 + 学习率0.001

---

#### 📍 模块5：结果对比

**功能说明**：综合对比传统ML与深度学习方法

**内容展示**：
1. **方法对比表格**：
   - K-means聚类：特点、适用场景、优势、局限
   - PCA降维：特点、适用场景、优势、局限
   - 神经网络：特点、适用场景、优势、局限

2. **本实验性能对比**：
   - 传统ML：轮廓系数、PCA解释方差
   - 深度学习：分类准确率、F1分数

3. **实验结论**：
   - 方法选择建议
   - 实际应用指导
   - 技术栈说明

---

### 三、实验建议流程

**推荐顺序**：
```
1. 数据采集（观察数据分布）
   ↓
2. 数据预处理（清洗+特征提取）
   ↓
3. K-means聚类（探索数据结构）
   ↓
4. PCA降维（可视化理解）
   ↓
5. 神经网络训练（精确分类）
   ↓
6. 结果对比（综合分析）
```

---

### 四、常见问题

**Q1：TensorFlow.js加载慢？**
- 首次加载需要下载模型库，约需5-10秒
- 后续使用会缓存，速度更快

**Q2：训练时间太长？**
- 降低训练轮数（10-20轮快速测试）
- 减少批量大小（16）

**Q3：准确率不高？**
- 增加训练轮数（50-100轮）
- 降低学习率（0.001）
- 检查数据预处理是否充分

**Q4：轮廓系数低？**
- 尝试调整K值（2-5）
- 检查特征向量质量
- PCA可能揭示数据分布问题

---

## 📊 评估指标说明

### 传统ML指标

| 指标 | 说明 | 范围 | 理想值 |
|------|------|------|--------|
| 轮廓系数 | 聚类质量评估 | 0-1 | >0.5 |
| 解释方差 | PCA信息保留量 | 0-100% | >70% |

### 深度学习指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| 准确率 | 正确预测比例 | (TP+TN)/(总数) |
| 精确率 | 预测为正中实际为正的比例 | TP/(TP+FP) |
| 召回率 | 实际为正中被预测为正的比例 | TP/(TP+FN) |
| F1分数 | 精确率和召回率的调和平均 | 2*P*R/(P+R) |

---

## 🔧 技术细节

### 依赖库版本
```json
{
  "@tensorflow/tfjs": "^4.22.0",
  "recharts": "^2.x",
  "mathjs": "^14.x",
  "papaparse": "^5.x"
}
```

### 运行环境要求
- Node.js 18+ 或 Node.js 24（推荐）
- pnpm 包管理器（必须）
- 现代浏览器（支持ES6+）

### 端口配置
- 开发环境：5000（固定）
- 热更新：自动启用

---

## 📝 项目特色

1. ✅ **完整流程**：涵盖数据挖掘全流程
2. ✅ **算法实现**：从原理实现K-means、PCA、神经网络
3. ✅ **可视化丰富**：10+种图表类型
4. ✅ **参数可调**：所有关键参数可实时调节
5. ✅ **教学导向**：清晰的算法说明和实验结论
6. ✅ **类型安全**：完整的TypeScript类型定义
7. ✅ **响应式设计**：支持桌面和移动设备

---

## 🎓 教学价值

本项目适合用于：
- 数据挖掘课程期末大作业
- 机器学习入门实践项目
- 深度学习入门教学案例
- 技术栈整合最佳实践展示

---

## 📞 使用支持

如有问题，可参考：
- `AGENTS.md` - 项目技术说明
- `DESIGN.md` - 设计规范文档
- 代码注释 - 每个函数都有详细说明

---

**祝你使用愉快！🎉**