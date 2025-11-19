#!/usr/bin/env python3
"""
AmbedkarGPT Evaluation Framework
Comprehensive RAG system evaluation with multiple metrics
"""

import warnings
import os
import json
import time
from typing import List, Dict, Tuple
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
CORPUS_DIR = "./corpus"
TEST_DATASET_PATH = "./test_dataset.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

# Chunking strategies to test
CHUNKING_STRATEGIES = {
    "small": {"chunk_size": 250, "chunk_overlap": 50},
    "medium": {"chunk_size": 550, "chunk_overlap": 75},
    "large": {"chunk_size": 900, "chunk_overlap": 100}
}


class RAGEvaluator:
    """Comprehensive RAG evaluation framework"""
    
    def __init__(self, corpus_dir: str, test_dataset_path: str):
        self.corpus_dir = corpus_dir
        self.test_dataset = self.load_test_dataset(test_dataset_path)
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.sentence_transformer = SentenceTransformer(EMBEDDING_MODEL)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.llm = Ollama(model=LLM_MODEL, temperature=0.3)
        
    def load_test_dataset(self, path: str) -> List[Dict]:
        """Load test dataset from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_questions']
    
    def load_corpus(self) -> List:
        """Load all documents from corpus"""
        import glob
        documents = []
        for file_path in glob.glob(os.path.join(self.corpus_dir, "*.txt")):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = os.path.basename(file_path)
            documents.extend(docs)
        return documents
    
    def create_chunks(self, documents: List, chunk_size: int, chunk_overlap: int) -> List:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, chunks: List, strategy_name: str) -> Chroma:
        """Create ChromaDB vector store"""
        chroma_dir = f"./chroma_db_{strategy_name}"
        
        # Remove old database
        if os.path.exists(chroma_dir):
            import shutil
            shutil.rmtree(chroma_dir)
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings_model,
            persist_directory=chroma_dir
        )
        return vector_store
    
    def setup_qa_chain(self, vector_store: Chroma, k: int = 4) -> RetrievalQA:
        """Setup QA chain with retriever"""
        prompt_template = """Use the following context to answer the question.
        If you cannot find the answer in the context, say "I cannot find this information in the provided documents."
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    
    # ==================== RETRIEVAL METRICS ====================
    
    def calculate_hit_rate(self, retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """
        Hit Rate: Did we retrieve at least one correct document?
        """
        if not expected_sources:  # Unanswerable question
            return 1.0
        
        retrieved_set = set(retrieved_sources)
        expected_set = set(expected_sources)
        
        return 1.0 if len(retrieved_set & expected_set) > 0 else 0.0
    
    def calculate_mrr(self, retrieved_sources: List[str], expected_sources: List[str]) -> float:
        """
        Mean Reciprocal Rank: Position of first relevant document
        """
        if not expected_sources:
            return 1.0
        
        expected_set = set(expected_sources)
        
        for rank, source in enumerate(retrieved_sources, 1):
            if source in expected_set:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_precision_at_k(self, retrieved_sources: List[str], expected_sources: List[str], k: int = 4) -> float:
        """
        Precision@K: Proportion of retrieved documents that are relevant
        """
        if not expected_sources:
            return 1.0
        
        retrieved_set = set(retrieved_sources[:k])
        expected_set = set(expected_sources)
        
        relevant_retrieved = len(retrieved_set & expected_set)
        return relevant_retrieved / min(k, len(retrieved_sources)) if retrieved_sources else 0.0
    
    # ==================== ANSWER QUALITY METRICS ====================
    
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Answer Relevance: Semantic similarity between question and answer
        """
        question_emb = self.sentence_transformer.encode([question])
        answer_emb = self.sentence_transformer.encode([answer])
        
        similarity = cosine_similarity(question_emb, answer_emb)[0][0]
        return float(similarity)
    
    def calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: Is answer grounded in retrieved context?
        Simple implementation: cosine similarity between answer and context
        """
        if not context.strip():
            return 0.0
        
        answer_emb = self.sentence_transformer.encode([answer])
        context_emb = self.sentence_transformer.encode([context])
        
        similarity = cosine_similarity(answer_emb, context_emb)[0][0]
        return float(similarity)
    
    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        """
        ROUGE-L Score: Longest common subsequence between generated and reference
        """
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    # ==================== SEMANTIC METRICS ====================
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Cosine Similarity: Semantic similarity between two texts
        """
        emb1 = self.sentence_transformer.encode([text1])
        emb2 = self.sentence_transformer.encode([text2])
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """
        BLEU Score: N-gram overlap between generated and reference
        """
        reference_tokens = reference.lower().split()
        generated_tokens = generated.lower().split()
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
        return score
    
    # ==================== EVALUATION PIPELINE ====================
    
    def evaluate_single_question(self, qa_chain: RetrievalQA, test_item: Dict) -> Dict:
        """Evaluate single question with all metrics"""
        question = test_item['question']
        ground_truth = test_item['ground_truth']
        expected_sources = test_item['source_documents']
        
        try:
            # Get answer from QA system
            result = qa_chain.invoke({"query": question})
            generated_answer = result['result']
            retrieved_docs = result.get('source_documents', [])
            
            # Extract source filenames
            retrieved_sources = [doc.metadata.get('source', '') for doc in retrieved_docs]
            
            # Extract context
            context = " ".join([doc.page_content for doc in retrieved_docs])
            
            # Calculate all metrics
            metrics = {
                # Retrieval Metrics
                "hit_rate": self.calculate_hit_rate(retrieved_sources, expected_sources),
                "mrr": self.calculate_mrr(retrieved_sources, expected_sources),
                "precision_at_k": self.calculate_precision_at_k(retrieved_sources, expected_sources, k=4),
                
                # Answer Quality Metrics
                "answer_relevance": self.calculate_answer_relevance(question, generated_answer),
                "faithfulness": self.calculate_faithfulness(generated_answer, context),
                "rouge_l": self.calculate_rouge_l(generated_answer, ground_truth),
                
                # Semantic Metrics
                "cosine_similarity": self.calculate_cosine_similarity(generated_answer, ground_truth),
                "bleu_score": self.calculate_bleu_score(generated_answer, ground_truth),
            }
            
            return {
                "question_id": test_item['id'],
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "metrics": metrics,
                "error": None
            }
            
        except Exception as e:
            return {
                "question_id": test_item['id'],
                "question": question,
                "error": str(e),
                "metrics": {}
            }
    
    def evaluate_strategy(self, strategy_name: str, chunk_size: int, chunk_overlap: int) -> Dict:
        """Evaluate RAG system with specific chunking strategy"""
        print(f"\n{'='*70}")
        print(f"Evaluating Strategy: {strategy_name.upper()}")
        print(f"Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"{'='*70}\n")
        
        # Load and chunk documents
        print("📄 Loading corpus...")
        documents = self.load_corpus()
        
        print(f"✂️  Creating chunks...")
        chunks = self.create_chunks(documents, chunk_size, chunk_overlap)
        print(f"   Created {len(chunks)} chunks")
        
        print("🔢 Creating vector store...")
        vector_store = self.create_vector_store(chunks, strategy_name)
        
        print("🤖 Setting up QA chain...")
        qa_chain = self.setup_qa_chain(vector_store)
        
        print(f"📝 Evaluating {len(self.test_dataset)} questions...\n")
        
        results = []
        for i, test_item in enumerate(self.test_dataset, 1):
            print(f"   Question {i}/{len(self.test_dataset)}: {test_item['question'][:60]}...")
            result = self.evaluate_single_question(qa_chain, test_item)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return {
            "strategy": strategy_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(chunks),
            "results": results,
            "aggregate_metrics": aggregate_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all questions"""
        valid_results = [r for r in results if r.get('metrics')]
        
        if not valid_results:
            return {}
        
        aggregate = {}
        metric_names = valid_results[0]['metrics'].keys()
        
        for metric_name in metric_names:
            values = [r['metrics'][metric_name] for r in valid_results]
            aggregate[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        # Calculate success rate (questions answered without error)
        total_questions = len(results)
        successful_questions = len(valid_results)
        aggregate['success_rate'] = successful_questions / total_questions
        
        return aggregate
    
    def run_full_evaluation(self) -> Dict:
        """Run evaluation across all chunking strategies"""
        print("\n" + "="*70)
        print("🚀 AMBEDKARGPT COMPREHENSIVE EVALUATION")
        print("="*70)
        
        all_results = {}
        
        for strategy_name, params in CHUNKING_STRATEGIES.items():
            strategy_results = self.evaluate_strategy(
                strategy_name,
                params['chunk_size'],
                params['chunk_overlap']
            )
            all_results[strategy_name] = strategy_results
        
        return all_results
    
    def save_results(self, results: Dict, output_path: str = "./test_results.json"):
        """Save evaluation results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: {output_path}")


def main():
    """Main execution function"""
    print("Initializing RAG Evaluator...")
    
    evaluator = RAGEvaluator(CORPUS_DIR, TEST_DATASET_PATH)
    
    print("Starting comprehensive evaluation...")
    results = evaluator.run_full_evaluation()
    
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    
    for strategy_name, strategy_data in results.items():
        agg = strategy_data['aggregate_metrics']
        print(f"\n{strategy_name.upper()} Strategy:")
        print(f"  Chunks: {strategy_data['num_chunks']}")
        print(f"  Success Rate: {agg['success_rate']:.2%}")
        print(f"  Hit Rate: {agg['hit_rate']['mean']:.3f}")
        print(f"  MRR: {agg['mrr']['mean']:.3f}")
        print(f"  ROUGE-L: {agg['rouge_l']['mean']:.3f}")
        print(f"  Cosine Similarity: {agg['cosine_similarity']['mean']:.3f}")
    
    evaluator.save_results(results)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
