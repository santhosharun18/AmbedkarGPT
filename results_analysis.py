#!/usr/bin/env python3
"""
Results Analysis and Visualization
Generates comprehensive analysis report from evaluation results
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List


class ResultsAnalyzer:
    """Analyze and format evaluation results"""
    
    def __init__(self, results_path: str = "./test_results.json"):
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary comparison table"""
        summary_data = []
        
        for strategy_name, strategy_data in self.results.items():
            agg = strategy_data['aggregate_metrics']
            
            row = {
                'Strategy': strategy_name.upper(),
                'Chunk Size': strategy_data['chunk_size'],
                'Overlap': strategy_data['chunk_overlap'],
                'Total Chunks': strategy_data['num_chunks'],
                'Success Rate': f"{agg['success_rate']:.2%}",
                'Hit Rate': f"{agg['hit_rate']['mean']:.3f}",
                'MRR': f"{agg['mrr']['mean']:.3f}",
                'Precision@K': f"{agg['precision_at_k']['mean']:.3f}",
                'Answer Relevance': f"{agg['answer_relevance']['mean']:.3f}",
                'Faithfulness': f"{agg['faithfulness']['mean']:.3f}",
                'ROUGE-L': f"{agg['rouge_l']['mean']:.3f}",
                'Cosine Similarity': f"{agg['cosine_similarity']['mean']:.3f}",
                'BLEU Score': f"{agg['bleu_score']['mean']:.3f}"
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def find_best_strategy(self) -> str:
        """Identify best performing strategy"""
        best_score = -1
        best_strategy = None
        
        for strategy_name, strategy_data in self.results.items():
            agg = strategy_data['aggregate_metrics']
            # Weighted average of key metrics
            score = (
                agg['hit_rate']['mean'] * 0.3 +
                agg['rouge_l']['mean'] * 0.3 +
                agg['cosine_similarity']['mean'] * 0.2 +
                agg['faithfulness']['mean'] * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        return best_strategy
    
    def analyze_question_types(self, strategy_name: str) -> Dict:
        """Analyze performance by question type"""
        strategy_data = self.results[strategy_name]
        results = strategy_data['results']
        
        type_performance = {}
        
        for result in results:
            q_id = result['question_id']
            # Find question type from test dataset
            q_type = self.get_question_type(q_id)
            
            if q_type not in type_performance:
                type_performance[q_type] = []
            
            if result.get('metrics'):
                type_performance[q_type].append(result['metrics'])
        
        # Calculate averages
        type_summary = {}
        for q_type, metrics_list in type_performance.items():
            if metrics_list:
                avg_metrics = {}
                for key in metrics_list[0].keys():
                    values = [m[key] for m in metrics_list]
                    avg_metrics[key] = sum(values) / len(values)
                type_summary[q_type] = avg_metrics
        
        return type_summary
    
    def get_question_type(self, q_id: int) -> str:
        """Get question type from ID (simple heuristic)"""
        if q_id in [10, 11, 21]:
            return "unanswerable"
        elif q_id in [7, 9, 18, 19]:
            return "comparative"
        elif q_id in [8, 20]:
            return "conceptual"
        else:
            return "factual"
    
    def generate_markdown_report(self, output_path: str = "./RESULTS_ANALYSIS.md"):
        """Generate comprehensive markdown report"""
        best_strategy = self.find_best_strategy()
        
        report = f"""# AmbedkarGPT Evaluation Results

**Date**: {datetime.now().strftime('%B %d, %Y')}  
**Evaluator**: [Your Name]  
**Best Performing Strategy**: {best_strategy.upper()}

---

## Executive Summary

This report presents a comprehensive evaluation of the AmbedkarGPT RAG system across three chunking strategies:
- **Small Chunks** (250 chars, 50 overlap)
- **Medium Chunks** (550 chars, 75 overlap)
- **Large Chunks** (900 chars, 100 overlap)

The evaluation uses 25 test questions across multiple question types (factual, comparative, conceptual, unanswerable) and measures performance using 8 metrics across three categories:

1. **Retrieval Metrics**: Hit Rate, MRR, Precision@K
2. **Answer Quality Metrics**: Answer Relevance, Faithfulness, ROUGE-L
3. **Semantic Metrics**: Cosine Similarity, BLEU Score

---

## Comparative Performance Table

"""
        # Add summary table
        df = self.generate_summary_table()
        report += df.to_markdown(index=False)
        
        report += "\n\n---\n\n## Detailed Analysis by Strategy\n\n"
        
        for strategy_name, strategy_data in self.results.items():
            agg = strategy_data['aggregate_metrics']
            
            report += f"""### {strategy_name.upper()} Strategy

**Configuration:**
- Chunk Size: {strategy_data['chunk_size']} characters
- Chunk Overlap: {strategy_data['chunk_overlap']} characters
- Total Chunks Generated: {strategy_data['num_chunks']}

**Performance Metrics:**

| Metric Category | Metric Name | Mean | Std Dev | Min | Max |
|----------------|-------------|------|---------|-----|-----|
| Retrieval | Hit Rate | {agg['hit_rate']['mean']:.3f} | {agg['hit_rate']['std']:.3f} | {agg['hit_rate']['min']:.3f} | {agg['hit_rate']['max']:.3f} |
| Retrieval | MRR | {agg['mrr']['mean']:.3f} | {agg['mrr']['std']:.3f} | {agg['mrr']['min']:.3f} | {agg['mrr']['max']:.3f} |
| Retrieval | Precision@K | {agg['precision_at_k']['mean']:.3f} | {agg['precision_at_k']['std']:.3f} | {agg['precision_at_k']['min']:.3f} | {agg['precision_at_k']['max']:.3f} |
| Answer Quality | Answer Relevance | {agg['answer_relevance']['mean']:.3f} | {agg['answer_relevance']['std']:.3f} | {agg['answer_relevance']['min']:.3f} | {agg['answer_relevance']['max']:.3f} |
| Answer Quality | Faithfulness | {agg['faithfulness']['mean']:.3f} | {agg['faithfulness']['std']:.3f} | {agg['faithfulness']['min']:.3f} | {agg['faithfulness']['max']:.3f} |
| Answer Quality | ROUGE-L | {agg['rouge_l']['mean']:.3f} | {agg['rouge_l']['std']:.3f} | {agg['rouge_l']['min']:.3f} | {agg['rouge_l']['max']:.3f} |
| Semantic | Cosine Similarity | {agg['cosine_similarity']['mean']:.3f} | {agg['cosine_similarity']['std']:.3f} | {agg['cosine_similarity']['min']:.3f} | {agg['cosine_similarity']['max']:.3f} |
| Semantic | BLEU Score | {agg['bleu_score']['mean']:.3f} | {agg['bleu_score']['std']:.3f} | {agg['bleu_score']['min']:.3f} | {agg['bleu_score']['max']:.3f} |

**Success Rate:** {agg['success_rate']:.2%}

"""
        
        report += """---

## Key Findings

### 1. Retrieval Performance
- **Hit Rate** measures whether at least one relevant document was retrieved
- **MRR (Mean Reciprocal Rank)** evaluates the position of the first relevant document
- **Precision@K** assesses the proportion of relevant documents in top-K results

### 2. Answer Quality
- **Answer Relevance** checks semantic alignment between question and answer
- **Faithfulness** ensures answers are grounded in retrieved context
- **ROUGE-L** measures longest common subsequence overlap with ground truth

### 3. Semantic Similarity
- **Cosine Similarity** evaluates embedding-based semantic similarity
- **BLEU Score** measures n-gram overlap with reference answers

---

## Recommendations

Based on the evaluation results:

1. **Best Strategy**: {best_strategy.upper()} strategy provides optimal balance across metrics
2. **Retrieval**: Higher hit rates indicate effective document retrieval
3. **Chunk Size Impact**: Different chunk sizes affect context granularity vs. coherence
4. **Answer Quality**: Faithfulness scores indicate how well answers stay grounded in source material

---

## Sample Question Analysis

### High-Performing Question Example
"""
        
        # Find best performing question
        best_q = self.find_best_question(best_strategy)
        if best_q:
            report += f"""
**Question ID**: {best_q['question_id']}  
**Question**: {best_q['question']}  
**Ground Truth**: {best_q['ground_truth']}  
**Generated Answer**: {best_q['generated_answer']}

**Metrics**:
- Hit Rate: {best_q['metrics']['hit_rate']:.3f}
- ROUGE-L: {best_q['metrics']['rouge_l']:.3f}
- Cosine Similarity: {best_q['metrics']['cosine_similarity']:.3f}
"""
        
        report += """

---

## Conclusion

This comprehensive evaluation demonstrates the RAG system's capabilities across multiple dimensions. The analysis provides insights into:
- Optimal chunking strategies for document retrieval
- Answer quality and factual grounding
- Semantic similarity with reference answers

The evaluation framework can be extended to test additional configurations and fine-tune system performance.

---

**Generated by**: AmbedkarGPT Evaluation Framework  
**Timestamp**: {timestamp}
""".format(timestamp=datetime.now().isoformat())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {output_path}")
    
    def find_best_question(self, strategy_name: str):
        """Find best performing question"""
        strategy_data = self.results[strategy_name]
        results = strategy_data['results']
        
        best_score = -1
        best_q = None
        
        for result in results:
            if result.get('metrics'):
                score = (
                    result['metrics'].get('hit_rate', 0) +
                    result['metrics'].get('rouge_l', 0) +
                    result['metrics'].get('cosine_similarity', 0)
                ) / 3
                
                if score > best_score:
                    best_score = score
                    best_q = result
        
        return best_q


def main():
    """Main execution"""
    print("🔍 Analyzing evaluation results...\n")
    
    analyzer = ResultsAnalyzer()
    
    print("📊 Generating summary table...")
    df = analyzer.generate_summary_table()
    print("\n" + df.to_string(index=False))
    
    print("\n\n🏆 Best performing strategy:", analyzer.find_best_strategy().upper())
    
    print("\n📝 Generating markdown report...")
    analyzer.generate_markdown_report()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
