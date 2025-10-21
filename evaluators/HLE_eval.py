import os
import re
import time
import gc
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

OPENAI_API_KEY = ""

MODEL = "gpt-5-mini"

DATA_PATHS = {
    "HLE": r"Your Path",
}

OUTPUT_DIR = r"Your Path"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 50
MAX_RETRIES = 2
BACKOFF_BASE = 2.0
MAX_INPUT_CHARS = 4000

client = OpenAI(api_key=OPENAI_API_KEY)

def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:\-–—!?\u3002\uFF0C]", "", s)
    return s

def get_category(subject: str) -> str:
    subject = str(subject).lower().strip()
    
    # Math
    if any(keyword in subject for keyword in [
        'math', 'calculus', 'algebra', 'geometry', 'statistics', 'probability',
        'number theory', 'combinatorics', 'topology', 'analysis', 'logic'
    ]):
        return 'Math'
    
    # Physics
    elif any(keyword in subject for keyword in [
        'physics', 'quantum', 'mechanics', 'thermodynamics', 'electromagnetism',
        'optics', 'relativity', 'particle', 'nuclear', 'astronomy', 'astrophysics'
    ]):
        return 'Physics'
    
    # Biology/Medicine
    elif any(keyword in subject for keyword in [
        'biology', 'medicine', 'medical', 'anatomy', 'physiology', 'genetics',
        'biochemistry', 'molecular biology', 'microbiology', 'immunology',
        'neuroscience', 'ecology', 'botany', 'zoology', 'pathology', 'pharmacology'
    ]):
        return 'Biology/Medicine'
    
    # Computer Science/Artificial Intelligence
    elif any(keyword in subject for keyword in [
        'computer science', 'artificial intelligence', 'machine learning',
        'programming', 'algorithms', 'data science', 'software', 'cybersecurity',
        'information technology', 'robotics', 'ai', 'ml', 'deep learning'
    ]):
        return 'Computer Science/Artificial Intelligence'
    
    # Engineering
    elif any(keyword in subject for keyword in [
        'engineering', 'mechanical', 'electrical', 'civil', 'chemical engineering',
        'aerospace', 'biomedical engineering', 'materials science', 'industrial'
    ]):
        return 'Engineering'
    
    # Chemistry
    elif any(keyword in subject for keyword in [
        'chemistry', 'organic chemistry', 'inorganic', 'physical chemistry',
        'analytical chemistry', 'chemical', 'electrochemistry'
    ]):
        return 'Chemistry'
    
    # Humanities/Social Science
    elif any(keyword in subject for keyword in [
        'humanities', 'social', 'history', 'literature', 'philosophy',
        'psychology', 'sociology', 'anthropology', 'economics', 'political',
        'linguistics', 'law', 'education', 'art', 'music', 'cultural'
    ]):
        return 'Humanities/Social Science'
    
    # Other
    else:
        return 'Other'

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
    df["category"] = df["subject"].apply(get_category)

    return df[["question","answer","subject","category","choices"]].dropna(subset=["question"]).reset_index(drop=True)

def make_prompt(q, choices=None):
    if isinstance(choices, (list,tuple)) and len(choices) > 0:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opts = "\n".join([f"{letters[i]}. {c}" for i,c in enumerate(choices[:len(letters)])])
        return f"You are an expert. Answer with ONLY one letter (A,B,C,...).\n\nQuestion:\n{q}\n\nChoices:\n{opts}\n\nAnswer:"
    return f"You are an expert. Provide ONLY the final short answer (no explanation).\n\nQuestion:\n{q}\n\nAnswer:"

def extract_letter(ans: str):
    m = re.search(r"\b([A-E])\b", ans or "", re.IGNORECASE)
    return m.group(1).upper() if m else ""

def ask_gpt(prompt: str):
    messages = [
        {"role": "system", "content": "You are a precise evaluator. Answer succinctly."},
        {"role": "user", "content": prompt}
    ]
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages)
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

def eval_dataset(name: str, df: pd.DataFrame):
    out_path = os.path.join(OUTPUT_DIR, f"{name}_gpt5mini.csv")
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
        preds, corrects = [], []

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"{name}[{beg}:{end})"):
            q = row["question"]
            a = row.get("answer", "")
            subj = row.get("subject", "unknown")
            cat = row.get("category", "Other")
            choices = row["choices"] if isinstance(row["choices"], (list,tuple)) else None
            prompt = make_prompt(q, choices)
            if len(prompt) > MAX_INPUT_CHARS:
                prompt = prompt[:MAX_INPUT_CHARS] + "\n...[truncated]"

            ans = ask_gpt(prompt)
            if ans.startswith("[Error:"):
                pred = ""
            elif choices:
                pred = extract_letter(ans)
            else:
                pred = ans.split("\n")[0].strip()

            score = 1.0 if norm(pred) == norm(a) else 0.0
            preds.append(pred)
            corrects.append(score)

        chunk["prediction"] = preds
        chunk["correct"] = corrects
        chunk.to_csv(out_path, mode="a" if beg else "w", index=False, header=not beg)
        gc.collect()

    full = pd.read_csv(out_path)
    acc = full["correct"].mean() if len(full) else 0.0
    per_sub = full.groupby("subject")["correct"].mean().sort_values(ascending=False)
    per_cat = full.groupby("category")["correct"].mean().sort_values(ascending=False)
    return acc, per_sub, per_cat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Choose one: HLE | HLE-Gold-Bio-Chem | BixBench")
    args = parser.parse_args()
    name = args.dataset.strip()

    if name not in DATA_PATHS:
        raise SystemExit(f"❌ Unknown dataset: {name}\nAvailable: {list(DATA_PATHS.keys())}")

    path = DATA_PATHS[name]
    if not os.path.exists(path):
        raise SystemExit(f"❌ File not found: {path}")

    print(f"\n===== Running {name} =====")
    df = load_any(path)
    acc, per_sub, per_cat = eval_dataset(name, df)
    print(f"[RESULT] {name}: ACC={acc*100:.2f}%")
    per_sub.to_csv(os.path.join(OUTPUT_DIR, f"{name}_per_subject.csv"))
    per_cat.to_csv(os.path.join(OUTPUT_DIR, f"{name}_per_category.csv"))
    print(f"[OK] Results saved to: {OUTPUT_DIR}")
    print(f"[OK] Per-subject accuracy saved to: {name}_per_subject.csv")
    print(f"[OK] Per-category accuracy saved to: {name}_per_category.csv")

if __name__ == "__main__":
    main()
