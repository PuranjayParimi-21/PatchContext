import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness
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
        
        try:
            res = rag.run(q)
            elapsed = time.perf_counter() - t_start
            logger.info(f"Completed in {elapsed:.2f}s. Answer: '{res['answer'][:100]}...'")
            
            questions.append(q)
            answers.append(res["answer"])
            contexts.append([doc.page_content for doc in res["retrieved_docs"]])
            ground_truths.append(gt)
            latencies.append(res["latencies"].get("total_response_latency", elapsed))
            confidence_scores.append(res["confidence_score"])
        except Exception as e:
            logger.error(f"Failed to generate answer for question {idx+1} ('{q}'): {e}", exc_info=True)
        
        # Respect OpenAI rate limits between queries
        time.sleep(1.0)
        
    # Evaluate each question individually to avoid one failure crashing the entire RAGAs run
    eval_results_list = []
    question_scores = []
    
    for i in range(len(questions)):
        single_dict = {
            "question": [questions[i]],
            "answer": [answers[i]],
            "contexts": [contexts[i]],
            "ground_truth": [ground_truths[i]]
        }
        try:
            single_dataset = Dataset.from_dict(single_dict)
            logger.info(f"[{i+1}/{len(questions)}] Running RAGAs metrics on question: '{questions[i]}'")
            res_eval = evaluate(
                dataset=single_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                    answer_correctness
                ]
            )
            eval_results_list.append(res_eval)
            question_scores.append(res_eval)
            logger.info(f"Successfully evaluated question {i+1} score: {res_eval}")
        except Exception as e:
            logger.error(f"Failed RAGAs evaluation on question {i+1} ('{questions[i]}'): {e}", exc_info=True)
            question_scores.append(None)
            
    completed_count = len(eval_results_list)
    mean_metrics = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
        "answer_correctness": 0.0
    }
    
    if completed_count > 0:
        for metric in mean_metrics.keys():
            total = sum(float(r.get(metric, 0.0)) for r in eval_results_list)
            mean_metrics[metric] = total / completed_count
            
    # Structure full report
    report = {
        "summary": {
            "num_questions": len(eval_questions),
            "completed_evaluations": completed_count,
            "evaluation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mean_metrics": mean_metrics,
            "avg_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
            "avg_nli_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        },
        "results": []
    }
    
    for i in range(len(questions)):
        scores_dict = question_scores[i] if (i < len(question_scores) and question_scores[i] is not None) else {}
        report["results"].append({
            "question": questions[i],
            "answer": answers[i],
            "ground_truth": ground_truths[i],
            "latency": latencies[i],
            "nli_confidence": confidence_scores[i],
            "contexts": contexts[i],
            "ragas_scores": {k: float(v) for k, v in scores_dict.items()}
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
        default=50, 
        help="Number of questions to evaluate (default: 50, range: 1 to 50)."
    )
    args = parser.parse_args()
    
    run_evaluation(args.limit)
