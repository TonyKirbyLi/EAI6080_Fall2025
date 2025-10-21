import os
import re
import time
import gc
import json
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from typing import List, Dict, Tuple, Optional
import random
import math

OPENAI_API_KEY = "Your Key"

MODEL = "gpt-5-mini" （Choose Your Model)

DATA_PATHS = {
    "HLE": r"Your Path",
}

OUTPUT_DIR = r"Your Path"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 5  
MAX_RETRIES = 3
BACKOFF_BASE = 2.0
MAX_INPUT_CHARS = 4000

TOT_BREADTH = 2 
TOT_DEPTH = 2  
TOT_EVALUATIONS = 1 

client = OpenAI(api_key=OPENAI_API_KEY)

def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:\-–—!?\u3002\uFF0C]", "", s)
    return s

def find_col(cols, keys):
    for k in keys:
        for c in cols:
            if k.lower() in str(c).lower():
                return c
    return None

def load_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unknown file format: {path}")

    q_col = find_col(df.columns, ["question","prompt","query","text","input"])
    a_col = find_col(df.columns, ["answer","final_answer","target","label"])
    s_col = find_col(df.columns, ["subject","domain","category","topic"])
    c_col = find_col(df.columns, ["choices","options"])

    if q_col is None:
        raise RuntimeError(f"No question column found in {path}")
    df = df.rename(columns={q_col:"question"})
    if a_col: df = df.rename(columns={a_col:"answer"})
    if s_col: df["subject"] = df[s_col]
    else: df["subject"] = "unknown"
    if c_col: df["choices"] = df[c_col]
    else: df["choices"] = None

    return df[["question","answer","subject","choices"]].dropna(subset=["question"]).reset_index(drop=True)

def ask_gpt(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise evaluator and expert problem solver."},
        {"role": "user", "content": prompt}
    ]
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL, 
                messages=messages
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError, APIStatusError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE ** attempt)
                continue
            return f"[Error: {e}]"
        except Exception as e:
            return f"[Error: {e}]"
    return f"[Error: {last_err}]" if last_err else "[Error: Unknown]"

def generate_thoughts(question: str, choices: Optional[List] = None, step: int = 1) -> List[str]:
    if isinstance(choices, (list, tuple)) and len(choices) > 0:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opts = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices[:len(letters)])])
        prompt = f"""Question: {question}

Choices:
{opts}

Generate {TOT_BREADTH} different reasoning approaches to solve this question. For each approach, provide:
1. A clear reasoning strategy
2. Key considerations
3. Step-by-step analysis

Format your response as:
APPROACH 1:
[reasoning approach 1]

APPROACH 2:
[reasoning approach 2]

APPROACH 3:
[reasoning approach 3]"""
    else:
        prompt = f"""Question: {question}

Generate {TOT_BREADTH} different reasoning approaches to solve this question. For each approach, provide:
1. A clear reasoning strategy
2. Key considerations  
3. Step-by-step analysis

Format your response as:
APPROACH 1:
[reasoning approach 1]

APPROACH 2:
[reasoning approach 2]

APPROACH 3:
[reasoning approach 3]"""

    response = ask_gpt(prompt)

    thoughts = []
    approaches = re.split(r'APPROACH \d+:', response)
    for approach in approaches[1:]:
        if approach.strip():
            thoughts.append(approach.strip())
    if not thoughts:
        thoughts = [response]
    
    return thoughts[:TOT_BREADTH]

class MCTSNode:
    def __init__(self, thought: str, parent=None):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_score = 0.0
        self.is_expanded = False
    
    def add_child(self, child_thought: str):
        child = MCTSNode(child_thought, self)
        self.children.append(child)
        return child
    
    def update(self, score: float):
        self.visits += 1
        self.total_score += score
    
    def get_average_score(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits
    
    def ucb1_score(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.get_average_score()
        
        exploitation = self.get_average_score()
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_best_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.ucb1_score())
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0

def mcts_evaluate_thoughts(thoughts: List[str], question: str, choices: Optional[List] = None, 
                          iterations: int = 50) -> List[Tuple[str, float]]:
    if not thoughts:
        return []
    
    root = MCTSNode("ROOT")
    
    thought_nodes = {}
    for thought in thoughts:
        child = root.add_child(thought)
        thought_nodes[thought] = child
    for _ in range(iterations):
        current = root
        path = [current]
        
        while not current.is_leaf():
            current = current.select_best_child()
            if current is None:
                break
            path.append(current)
        
        if current is None:
            continue
        
        if current.thought != "ROOT":
            score = evaluate_thought_simple(current.thought, question, choices)
            normalized_score = (score - 1.0) / 9.0
        else:
            normalized_score = 0.5
        for node in reversed(path):
            node.update(normalized_score)

    results = []
    for thought in thoughts:
        node = thought_nodes[thought]
        avg_score = node.get_average_score()
        final_score = avg_score * 9.0 + 1.0
        results.append((thought, final_score))
    
    return results

def evaluate_thought_mcts(thought: str, question: str, choices: Optional[List] = None) -> float:
    thoughts = [thought]
    results = mcts_evaluate_thoughts(thoughts, question, choices, iterations=20)
    if results:
        return results[0][1]
    return 5.0

def evaluate_thought_simple(thought: str, question: str, choices: Optional[List] = None) -> float:
    score = 5.0 
    thought_len = len(thought.strip())
    if thought_len > 50:
        score += 1.0
    if thought_len > 150:
        score += 0.5
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    thought_words = set(re.findall(r'\b\w+\b', thought.lower()))

    overlap = len(question_words.intersection(thought_words))
    if overlap > 3:
        score += 1.0
    elif overlap > 1:
        score += 0.5
    reasoning_words = ['because', 'therefore', 'since', 'thus', 'hence', 'so', 'if', 'then', 
                      'first', 'second', 'next', 'finally', 'however', 'but', 'although']
    reasoning_count = sum(1 for word in reasoning_words if word in thought.lower())
    if reasoning_count > 2:
        score += 1.0
    elif reasoning_count > 0:
        score += 0.5

    if choices and isinstance(choices, (list, tuple)):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        mentioned_options = sum(1 for i, choice in enumerate(choices[:len(letters)]) 
                               if letters[i] in thought or str(choice).lower() in thought.lower())
        if mentioned_options > 0:
            score += 0.5
    if any(marker in thought.lower() for marker in ['step', 'analyze', 'consider', 'examine']):
        score += 0.5
    return max(1.0, min(10.0, score))

def evaluate_thought(thought: str, question: str, choices: Optional[List] = None) -> float:
    return evaluate_thought_mcts(thought, question, choices)

def select_best_thoughts(thoughts: List[str], question: str, choices: Optional[List] = None, top_k: int = 2) -> List[str]:
    if len(thoughts) <= top_k:
        return thoughts
    thought_scores = mcts_evaluate_thoughts(thoughts, question, choices, iterations=100)

    thought_scores.sort(key=lambda x: x[1], reverse=True)
    return [thought for thought, score in thought_scores[:top_k]]

def generate_final_answer(best_thoughts: List[str], question: str, choices: Optional[List] = None) -> str:
    combined_reasoning = "\n\n".join([f"Reasoning {i+1}:\n{thought}" for i, thought in enumerate(best_thoughts)])
    
    if isinstance(choices, (list, tuple)) and len(choices) > 0:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opts = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices[:len(letters)])])
        prompt = f"""Question: {question}

Choices:
{opts}

Based on the following reasoning approaches:
{combined_reasoning}

Synthesize these approaches and provide your final answer. Answer with ONLY one letter (A,B,C,...)."""
    else:
        prompt = f"""Question: {question}

Based on the following reasoning approaches:
{combined_reasoning}

Synthesize these approaches and provide your final answer. Provide ONLY the final short answer (no explanation)."""

    return ask_gpt(prompt)

def tot_solve(question: str, choices: Optional[List] = None) -> Tuple[str, Dict]:
    tot_log = {
        "steps": [],
        "final_reasoning": [],
        "total_api_calls": 0,
        "mcts_iterations": 0
    }
    
    print(f"  Generate reasoning path...")
    thoughts = generate_thoughts(question, choices, 1)
    tot_log["total_api_calls"] += len(thoughts)
    
    if len(thoughts) > 1:
        print(f"  Use MCTS eval {len(thoughts)} reasoning path...")
        best_thoughts = select_best_thoughts(thoughts, question, choices, top_k=1)
        tot_log["mcts_iterations"] = 100
    else:
        best_thoughts = thoughts
    
    tot_log["steps"].append({
        "step": 1,
        "generated_thoughts": len(thoughts),
        "selected_thoughts": len(best_thoughts),
        "best_thoughts": best_thoughts
    })
    
    print(f"  Generate Fianl answer...")
    final_answer = generate_final_answer(best_thoughts, question, choices)
    tot_log["total_api_calls"] += 1
    tot_log["final_reasoning"] = best_thoughts
    
    return final_answer, tot_log

def extract_letter(ans: str):
    m = re.search(r"\b([A-E])\b", ans or "", re.IGNORECASE)
    return m.group(1).upper() if m else ""

def eval_dataset_with_tot(name: str, df: pd.DataFrame):
    out_path = os.path.join(OUTPUT_DIR, f"{name}_gpt5mini_tot.csv")
    log_path = os.path.join(OUTPUT_DIR, f"{name}_gpt5mini_tot_logs.jsonl")
    
    done = 0
    if os.path.exists(out_path):
        try:
            done = len(pd.read_csv(out_path))
        except Exception:
            done = 0

    n = len(df)
    for beg in range(done, n, CHUNK_SIZE):
        end = min(beg + CHUNK_SIZE, n)
        chunk = df.iloc[beg:end].copy()
        preds, corrects, tot_logs = [], [], []

        for idx, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"{name}[{beg}:{end}] TOT"):
            q = row["question"]
            a = row.get("answer", "")
            subj = row.get("subject", "unknown")
            choices = row["choices"] if isinstance(row["choices"], (list, tuple)) else None
            
            print(f"\n Deal question {idx}: {q[:100]}...")
            if len(q) > MAX_INPUT_CHARS:
                q = q[:MAX_INPUT_CHARS] + "\n...[truncated]"

            try:
                ans, tot_log = tot_solve(q, choices)
                
                if ans.startswith("[Error:"):
                    pred = ""
                elif choices:
                    pred = extract_letter(ans)
                else:
                    pred = ans.split("\n")[0].strip()

                score = 1.0 if norm(pred) == norm(a) else 0.0
                print(f"  Pred: {pred}, Answer: {a}, Score: {score}")

                tot_log.update({
                    "question_id": idx,
                    "question": q,
                    "true_answer": a,
                    "predicted_answer": pred,
                    "correct": score,
                    "subject": subj
                })

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(tot_log, ensure_ascii=False) + "\n")
                
            except Exception as e:
                pred = f"[Error: {e}]"
                score = 0.0
                print(f" Deal question {idx} error: {e}")

            preds.append(pred)
            corrects.append(score)

        chunk["prediction"] = preds
        chunk["correct"] = corrects
        chunk.to_csv(out_path, mode="a" if beg else "w", index=False, header=not beg)
        gc.collect()

    full = pd.read_csv(out_path)
    acc = full["correct"].mean() if len(full) else 0.0
    per_sub = full.groupby("subject")["correct"].mean().sort_values(ascending=False)
    
    return acc, per_sub

def main():
    parser = argparse.ArgumentParser(description="Evaluate datasets using Tree of Thoughts with GPT-5-mini")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Choose one: HLE")
    parser.add_argument("--breadth", type=int, default=3,
                        help="Number of thoughts to generate at each step")
    parser.add_argument("--depth", type=int, default=3,
                        help="Depth of the thought tree")
    parser.add_argument("--evaluations", type=int, default=3,
                        help="Number of evaluations per thought")
    args = parser.parse_args()

    global TOT_BREADTH, TOT_DEPTH, TOT_EVALUATIONS
    TOT_BREADTH = args.breadth
    TOT_DEPTH = args.depth
    TOT_EVALUATIONS = args.evaluations
    
    name = args.dataset.strip()

    if name not in DATA_PATHS:
        raise SystemExit(f"❌ Unknown dataset: {name}\nAvailable: {list(DATA_PATHS.keys())}")

    path = DATA_PATHS[name]
    if not os.path.exists(path):
        raise SystemExit(f"❌ File not found: {path}")

    print(f"\n===== Running {name} with Tree of Thoughts =====")
    print(f"TOT Parameters: Breadth={TOT_BREADTH}, Depth={TOT_DEPTH}, Evaluations={TOT_EVALUATIONS}")
    
    df = load_any(path)
    print(f"Loaded {len(df)} questions")
    
    acc, per = eval_dataset_with_tot(name, df)
    print(f"[RESULT] {name} with TOT: ACC={acc*100:.2f}%")

    per.to_csv(os.path.join(OUTPUT_DIR, f"{name}_tot_per_subject.csv"))
    print(f"[OK] Results saved to: {OUTPUT_DIR}")
    print(f"[OK] Detailed TOT logs saved to: {os.path.join(OUTPUT_DIR, f'{name}_gpt5mini_tot_logs.jsonl')}")

if __name__ == "__main__":
    main()
