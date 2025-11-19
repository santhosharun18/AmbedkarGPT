# AmbedkarGPT Evaluation Results

**Date**: November 17, 2025  
**Evaluator**: [Your Name]  
**Best Performing Strategy**: SMALL

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

| Strategy   |   Chunk Size |   Overlap |   Total Chunks | Success Rate   |   Hit Rate |   MRR |   Precision@K |   Answer Relevance |   Faithfulness |   ROUGE-L |   Cosine Similarity |   BLEU Score |
|:-----------|-------------:|----------:|---------------:|:---------------|-----------:|------:|--------------:|-------------------:|---------------:|----------:|--------------------:|-------------:|
| SMALL      |          250 |        50 |             26 | 100.00%        |          1 | 0.903 |          0.38 |              0.723 |          0.585 |     0.33  |               0.579 |        0.107 |
| MEDIUM     |          550 |        75 |             13 | 100.00%        |          1 | 0.907 |          0.39 |              0.737 |          0.555 |     0.348 |               0.577 |        0.116 |
| LARGE      |          900 |       100 |              7 | 100.00%        |          1 | 0.913 |          0.39 |              0.708 |          0.55  |     0.317 |               0.587 |        0.094 |

---

## Detailed Analysis by Strategy

### SMALL Strategy

**Configuration:**
- Chunk Size: 250 characters
- Chunk Overlap: 50 characters
- Total Chunks Generated: 26

**Performance Metrics:**

| Metric Category | Metric Name | Mean | Std Dev | Min | Max |
|----------------|-------------|------|---------|-----|-----|
| Retrieval | Hit Rate | 1.000 | 0.000 | 1.000 | 1.000 |
| Retrieval | MRR | 0.903 | 0.226 | 0.250 | 1.000 |
| Retrieval | Precision@K | 0.380 | 0.246 | 0.250 | 1.000 |
| Answer Quality | Answer Relevance | 0.723 | 0.189 | 0.067 | 0.915 |
| Answer Quality | Faithfulness | 0.585 | 0.227 | 0.003 | 0.859 |
| Answer Quality | ROUGE-L | 0.330 | 0.198 | 0.073 | 0.722 |
| Semantic | Cosine Similarity | 0.579 | 0.137 | 0.349 | 0.847 |
| Semantic | BLEU Score | 0.107 | 0.116 | 0.004 | 0.355 |

**Success Rate:** 100.00%

### MEDIUM Strategy

**Configuration:**
- Chunk Size: 550 characters
- Chunk Overlap: 75 characters
- Total Chunks Generated: 13

**Performance Metrics:**

| Metric Category | Metric Name | Mean | Std Dev | Min | Max |
|----------------|-------------|------|---------|-----|-----|
| Retrieval | Hit Rate | 1.000 | 0.000 | 1.000 | 1.000 |
| Retrieval | MRR | 0.907 | 0.216 | 0.333 | 1.000 |
| Retrieval | Precision@K | 0.390 | 0.246 | 0.250 | 1.000 |
| Answer Quality | Answer Relevance | 0.737 | 0.135 | 0.189 | 0.888 |
| Answer Quality | Faithfulness | 0.555 | 0.208 | -0.007 | 0.857 |
| Answer Quality | ROUGE-L | 0.348 | 0.194 | 0.103 | 0.837 |
| Semantic | Cosine Similarity | 0.577 | 0.140 | 0.280 | 0.850 |
| Semantic | BLEU Score | 0.116 | 0.143 | 0.004 | 0.632 |

**Success Rate:** 100.00%

### LARGE Strategy

**Configuration:**
- Chunk Size: 900 characters
- Chunk Overlap: 100 characters
- Total Chunks Generated: 7

**Performance Metrics:**

| Metric Category | Metric Name | Mean | Std Dev | Min | Max |
|----------------|-------------|------|---------|-----|-----|
| Retrieval | Hit Rate | 1.000 | 0.000 | 1.000 | 1.000 |
| Retrieval | MRR | 0.913 | 0.201 | 0.333 | 1.000 |
| Retrieval | Precision@K | 0.390 | 0.246 | 0.250 | 1.000 |
| Answer Quality | Answer Relevance | 0.708 | 0.127 | 0.178 | 0.869 |
| Answer Quality | Faithfulness | 0.550 | 0.209 | 0.017 | 0.814 |
| Answer Quality | ROUGE-L | 0.317 | 0.175 | 0.100 | 0.722 |
| Semantic | Cosine Similarity | 0.587 | 0.139 | 0.280 | 0.847 |
| Semantic | BLEU Score | 0.094 | 0.110 | 0.008 | 0.355 |

**Success Rate:** 100.00%

---

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

**Question ID**: 10  
**Question**: What was Ambedkar's favorite food?  
**Ground Truth**: This information is not available in the provided documents.  
**Generated Answer**:  I cannot find this information in the provided documents.

**Metrics**:
- Hit Rate: 1.000
- ROUGE-L: 0.667
- Cosine Similarity: 0.847


---

## Conclusion

This comprehensive evaluation demonstrates the RAG system's capabilities across multiple dimensions. The analysis provides insights into:
- Optimal chunking strategies for document retrieval
- Answer quality and factual grounding
- Semantic similarity with reference answers

The evaluation framework can be extended to test additional configurations and fine-tune system performance.

---

**Generated by**: AmbedkarGPT Evaluation Framework  
**Timestamp**: 2025-11-17T22:43:19.509530
