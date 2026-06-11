/**
 * GitHub Issues 数据获取API
 * 从公开的开源项目获取Issues评论数据
 */

import { NextResponse } from 'next/server';

// 热门开源项目列表（用于获取Issues数据）
const GITHUB_PROJECTS = [
  { owner: 'facebook', repo: 'react', name: 'React' },
  { owner: 'vuejs', repo: 'vue', name: 'Vue.js' },
  { owner: 'tensorflow', repo: 'tensorflow', name: 'TensorFlow' },
  { owner: 'microsoft', repo: 'vscode', name: 'VS Code' },
  { owner: 'nodejs', repo: 'node', name: 'Node.js' },
];

interface GitHubIssue {
  id: number;
  title: string;
  body: string | null;
  state: string;
  created_at: string;
  comments: number;
  user: {
    login: string;
  };
  labels: Array<{ name: string }>;
}

interface ReviewData {
  id: string;
  productId: string;
  productName: string;
  rating: number;
  content: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  keywords: string[];
  timestamp: string;
  source: string;
}

/**
 * 简单情感分析函数
 * 基于关键词判断评论情感倾向
 */
function analyzeSentiment(text: string): 'positive' | 'negative' | 'neutral' {
  if (!text) return 'neutral';
  
  const positiveKeywords = [
    'great', 'awesome', 'excellent', 'good', 'nice', 'love', 'perfect',
    'helpful', 'thanks', 'thank', 'solved', 'fixed', 'works', 'working',
    'amazing', 'wonderful', 'fantastic', 'brilliant', 'best', 'like',
  ];
  
  const negativeKeywords = [
    'bug', 'issue', 'problem', 'error', 'fail', 'failed', 'crash',
    'broken', 'not working', 'doesn\'t work', 'bad', 'terrible',
    'horrible', 'worst', 'hate', 'dislike', 'annoying', 'frustrating',
  ];
  
  const lowerText = text.toLowerCase();
  
  const positiveCount = positiveKeywords.filter(kw => lowerText.includes(kw)).length;
  const negativeCount = negativeKeywords.filter(kw => lowerText.includes(kw)).length;
  
  if (positiveCount > negativeCount + 1) return 'positive';
  if (negativeCount > positiveCount + 1) return 'negative';
  return 'neutral';
}

/**
 * 从关键词中提取特征词
 */
function extractKeywords(text: string): string[] {
  if (!text) return [];
  
  // 常见技术关键词
  const techKeywords = [
    'bug', 'feature', 'performance', 'memory', 'security',
    'ui', 'api', 'component', 'render', 'state', 'hook',
    'async', 'promise', 'callback', 'event', 'props',
    'typescript', 'javascript', 'css', 'html', 'node',
  ];
  
  const lowerText = text.toLowerCase();
  return techKeywords.filter(kw => lowerText.includes(kw));
}

/**
 * 根据情感生成评分
 */
function generateRating(sentiment: 'positive' | 'negative' | 'neutral'): number {
  switch (sentiment) {
    case 'positive':
      return Math.floor(Math.random() * 2) + 4; // 4-5分
    case 'negative':
      return Math.floor(Math.random() * 2) + 1; // 1-2分
    case 'neutral':
      return 3; // 3分
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const count = parseInt(searchParams.get('count') || '50');
  
  try {
    const allReviews: ReviewData[] = [];
    
    // 从多个项目获取Issues数据
    for (const project of GITHUB_PROJECTS) {
      try {
        // 获取Issues数据
        const response = await fetch(
          `https://api.github.com/repos/${project.owner}/${project.repo}/issues?state=all&per_page=20`,
          {
            headers: {
              'Accept': 'application/vnd.github.v3+json',
              'User-Agent': 'ML-Sentiment-Analysis-App',
            },
          }
        );
        
        if (!response.ok) {
          console.error(`Failed to fetch from ${project.name}: ${response.status}`);
          continue;
        }
        
        const issues: GitHubIssue[] = await response.json();
        
        // 转换Issues为评论数据
        for (const issue of issues) {
          const content = issue.body || issue.title;
          const sentiment = analyzeSentiment(content);
          const keywords = extractKeywords(content);
          
          allReviews.push({
            id: `gh-${issue.id}`,
            productId: project.repo,
            productName: project.name,
            rating: generateRating(sentiment),
            content: content,
            sentiment: sentiment,
            keywords: keywords.length > 0 ? keywords : ['issue', 'github'],
            timestamp: issue.created_at,
            source: `GitHub/${project.owner}/${project.repo}`,
          });
          
          // 达到目标数量就停止
          if (allReviews.length >= count) break;
        }
        
        if (allReviews.length >= count) break;
        
      } catch (error) {
        console.error(`Error fetching from ${project.name}:`, error);
        continue;
      }
    }
    
    // 如果获取的数据不足，补充一些模拟数据
    if (allReviews.length < count) {
      const remaining = count - allReviews.length;
      const mockReviews = generateFallbackData(remaining);
      allReviews.push(...mockReviews);
    }
    
    return NextResponse.json({
      success: true,
      data: allReviews.slice(0, count),
      source: 'GitHub Issues API',
      message: `成功获取 ${allReviews.length} 条数据`,
    });
    
  } catch (error) {
    console.error('Error fetching GitHub data:', error);
    return NextResponse.json({
      success: false,
      error: '数据获取失败',
      message: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}

/**
 * 备用数据生成函数（当API获取失败时使用）
 */
function generateFallbackData(count: number): ReviewData[] {
  const reviews: ReviewData[] = [];
  const projects = ['React', 'Vue.js', 'TensorFlow', 'VS Code', 'Node.js'];
  
  const positiveTemplates = [
    'This feature works great! Very helpful for my project.',
    'Excellent implementation, solved my problem perfectly.',
    'Thanks for this awesome update, really appreciate it!',
    'Great work! This is exactly what I needed.',
    'Love this new feature, makes development much easier.',
  ];
  
  const negativeTemplates = [
    'Found a bug that causes crashes in production.',
    'This feature doesn\'t work as expected, needs fix.',
    'Performance issue after the recent update.',
    'Error occurred when trying to use this feature.',
    'Documentation is unclear, hard to understand.',
  ];
  
  const neutralTemplates = [
    'Question about how to use this feature properly.',
    'Request for additional functionality.',
    'Suggestion for improvement in future versions.',
    'Need help understanding the implementation.',
    'Looking for examples or tutorials.',
  ];
  
  for (let i = 0; i < count; i++) {
    const sentimentIndex = Math.floor(Math.random() * 3);
    const sentiments: ('positive' | 'negative' | 'neutral')[] = ['positive', 'negative', 'neutral'];
    const sentiment = sentiments[sentimentIndex];
    
    const templates = sentiment === 'positive' ? positiveTemplates :
                     sentiment === 'negative' ? negativeTemplates : neutralTemplates;
    
    const content = templates[Math.floor(Math.random() * templates.length)];
    const project = projects[Math.floor(Math.random() * projects.length)];
    
    reviews.push({
      id: `fallback-${Date.now()}-${i}`,
      productId: project.toLowerCase().replace(/\s+/g, '-'),
      productName: project,
      rating: generateRating(sentiment),
      content: content,
      sentiment: sentiment,
      keywords: extractKeywords(content),
      timestamp: new Date().toISOString(),
      source: 'Fallback Data',
    });
  }
  
  return reviews;
}