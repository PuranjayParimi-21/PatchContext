import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from app.config import settings
from app.database import DatabaseManager
from app.rag_pipeline import PatchContextRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PatchContext.Evaluator")

def run_evaluation(num_questions: int = 5) -> None:
    """Runs RAGAs evaluation on a subset (or all) of the benchmark questions."""
    logger.info(f"Starting RAGAs evaluation on top {num_questions} questions...")
    
    # Check OpenAI API Key
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required to run RAGAs evaluations.")
        return
        
    # Check questions file
    questions_path = "benchmark/questions.json"
    if not os.path.exists(questions_path):
        logger.error(f"Questions file not found at {questions_path}")
        return
        
    with open(questions_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
        
    # Limit number of questions
    eval_questions = all_questions[:num_questions]
    logger.info(f"Loaded {len(eval_questions)} questions for evaluation.")
    
    # Initialize DB and Pipeline
    db = DatabaseManager(settings.database_path)
    rag = PatchContextRAG(db)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    latencies = []
    confidence_scores = []
    
    # Run the pipeline on each question
    for idx, q_item in enumerate(eval_questions):
        q = q_item["question"]
        gt = q_item["ground_truth"]
        
        logger.info(f"[{idx+1}/{len(eval_questions)}] Evaluating query: '{q}'")
        t_start = time.perf_counter()
        
        res = rag.run(q)
        
        elapsed = time.perf_counter() - t_start
        logger.info(f"Completed in {elapsed:.2f}s. Answer: '{res['answer'][:100]}...'")
        
        questions.append(q)
        answers.append(res["answer"])
        # RAGAs expects contexts as List[List[str]], where each inner list contains retrieved text fragments
        contexts.append([doc.page_content for doc in res["retrieved_docs"]])
        ground_truths.append(gt)
        latencies.append(res["latencies"].get("total_response_latency", elapsed))
        confidence_scores.append(res["confidence_score"])
        
        # Respect OpenAI rate limits between queries
        time.sleep(1.0)
        
    # Build datasets.Dataset
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    # Fallback when RAGAs run fails (e.g. rate limit, library issues, etc.)
    eval_results = {}
    try:
        dataset = Dataset.from_dict(dataset_dict)
        logger.info("Invoking RAGAs evaluation metrics...")
        
        # Override OpenAI model settings if needed, by default RAGAs uses langchain's chat openai
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ]
        )
        
        eval_results = result
        logger.info("Evaluation completed successfully.")
        logger.info(f"RAGAs Mean Scores:\n{result}")
    except Exception as e:
        logger.error(f"RAGAs evaluation failed or encountered rate limits: {e}", exc_info=True)
        # Initialize mock/zero results to save run data safely
        eval_results = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0
        }
        
    # Structure full report
    report = {
        "summary": {
            "num_questions": len(eval_questions),
            "evaluation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mean_metrics": {k: float(v) for k, v in eval_results.items()} if hasattr(eval_results, "items") else eval_results,
            "avg_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
            "avg_nli_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        },
        "results": []
    }
    
    for i in range(len(questions)):
        report["results"].append({
            "question": questions[i],
            "answer": answers[i],
            "ground_truth": ground_truths[i],
            "latency": latencies[i],
            "nli_confidence": confidence_scores[i],
            "contexts": contexts[i]
        })
        
    os.makedirs("benchmark", exist_ok=True)
    answers_path = "benchmark/answers.json"
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved evaluation report to {answers_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PatchContext RAG pipeline using RAGAs.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=5, 
        help="Number of questions to evaluate (default: 5, range: 1 to 50)."
    )
    args = parser.parse_args()
    
    run_evaluation(args.limit)
