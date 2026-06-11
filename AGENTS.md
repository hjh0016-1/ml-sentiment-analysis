# AGENTS.md

## 项目概览

**项目名称**：数据挖掘与机器学习实践平台 - 电商产品评论情感分析

**项目类型**：期末大作业项目，用于展示完整的数据挖掘与机器学习工作流程

**主要功能**：
- 数据采集：模拟爬虫采集电商产品评论数据
- 数据预处理：数据清洗、转换、特征提取
- 传统机器学习：K-means聚类、PCA降维
- 深度学习：神经网络情感分类模型
- 结果可视化：交互式图表展示与对比分析

## 技术栈

- **Framework**: Next.js 16 (App Router)
- **Core**: React 19
- **Language**: TypeScript 5
- **UI Components**: shadcn/ui (基于 Radix UI)
- **Styling**: Tailwind CSS 4
- **Visualization**: Recharts
- **Deep Learning**: TensorFlow.js
- **Math**: mathjs

## 核心模块

### 1. 数据采集模块 (src/lib/data-processing.ts)
- `generateMockReviews()` - 生成模拟电商评论数据
- 支持正向、负向、中性三种情感标签
- 包含评分、评论内容、关键词等特征

### 2. 数据预处理模块 (src/lib/data-processing.ts)
- `calculateDataStats()` - 计算数据统计信息
- `preprocessData()` - 数据清洗（去重、去缺失值、去异常值）
- `extractFeatures()` - 特征提取（15维特征向量）
- `normalizeFeatures()` - 特征归一化（Z-score）

### 3. 机器学习模块 (src/lib/ml-algorithms.ts)
- `kMeansClustering()` - K-means聚类算法实现
- `performPCA()` - PCA降维算法实现
- `calculateSilhouetteScore()` - 聚类质量评估（轮廓系数）

### 4. 深度学习模块 (src/lib/dl-model.ts)
- `NeuralNetworkClassifier` - 神经网络分类器类
- 支持多层网络架构配置
- 使用 TensorFlow.js 实现训练与预测
- 包含训练历史记录和模型评估指标

### 5. 主界面 (src/app/page.tsx)
- Tab式布局：数据采集、预处理、ML、DL、对比分析
- 实时参数调节（滑块控制）
- 动态可视化图表（Recharts）
- 进度展示和状态反馈

## 构建和测试命令

```bash
# 安装依赖
pnpm install

# 开发环境
pnpm dev

# TypeScript 类型检查
pnpm ts-check

# ESLint 检查
pnpm lint:build --quiet

# 构建
pnpm build

# 生产环境启动
pnpm start
```

## 代码风格指南

- 使用 TypeScript strict 模式
- 遵循 Next.js App Router 规范
- 使用 shadcn/ui 组件库
- 响应式设计（支持移动端）
- 清晰的类型定义和注释

## 目录结构

```
src/
├── app/
│   ├── layout.tsx        # 应用布局
│   ├── page.tsx          # 主页面（所有功能模块）
│   └── globals.css       # 全局样式
├── components/
│   └── ui/               # shadcn/ui 组件库
├── hooks/
│   └── use-mobile.ts     # 移动端检测
├── lib/
│   ├── data-processing.ts  # 数据处理工具
│   ├── ml-algorithms.ts    # ML算法实现
│   ├── dl-model.ts         # DL模型实现
│   └── utils.ts            # 通用工具函数
├── types/
│   └── index.ts           # 类型定义
└── server.ts              # 服务端配置
```

## 关键算法说明

### K-means聚类
- 支持欧氏距离和余弦距离
- 自动迭代直到收敛
- 返回聚类中心、聚类标签、统计信息
- 使用轮廓系数评估聚类质量

### PCA降维
- 计算协方差矩阵和特征值
- 选择最大方差方向作为主成分
- 返回降维数据、解释方差比例
- 支持可视化展示（2D散点图）

### 神经网络
- 多层架构：输入层(15维) → 隐藏层(64-32-16) → 输出层(3分类)
- Dropout层防止过拟合
- Adam优化器 + 分类交叉熵损失
- 训练历史追踪（损失、准确率）

## 数据流程

1. **采集**：生成200条模拟评论数据
2. **预处理**：清洗、提取15维特征、归一化
3. **ML分析**：K-means聚类 + PCA降维可视化
4. **DL训练**：划分训练集/测试集，训练神经网络
5. **评估**：计算准确率、精确率、召回率、F1分数

## 可视化类型

- 饼图：情感分布展示
- 柱状图：评分分布、混淆矩阵
- 折线图：训练损失和准确率曲线
- 散点图：PCA降维后的聚类可视化
- 进度条：爬虫采集、模型训练进度

## 注意事项

- TensorFlow.js 在浏览器端运行，首次加载可能较慢
- 训练时间取决于epochs设置和数据量
- 轮廓系数越高，聚类质量越好
- F1分数综合考虑精确率和召回率

## 教学价值

本项目展示了：
1. 完整的数据挖掘工作流程
2. 传统ML与DL方法的对比
3. 数据可视化的重要性
4. 模型评估指标的解读
5. 技术栈整合的最佳实践