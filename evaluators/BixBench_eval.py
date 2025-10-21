import os
import re
import io
import gc
import json
import time
import base64
import zipfile
import mimetypes
import argparse
from collections import defaultdict, OrderedDict

import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

MAX_IMAGES_PER_SAMPLE = 4         
MAX_RETRIES = 3                  
BACKOFF_BASE = 2.0             
CHUNK_SIZE = 50   
MAX_OUTPUT_TOKENS = 128  
TIMEOUT_HINT = 15

def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:\-–—!?\u3002\uFF0C]", "", s)
    return s

def load_bixbench(jsonl_path: str) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)
    q_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["question","input","query","prompt"])), None)
    if not q_col:
        raise RuntimeError("❌ JSONL No question/input raw")
    df = df.rename(columns={q_col: "question"})
    gold_keys = ["hypothesis", "answer", "label", "target", "final_answer"]
    a_col = next((c for c in df.columns if any(k == str(c).lower() or k in str(c).lower() for k in gold_keys)), None)
    if a_col:
        df = df.rename(columns={a_col: "gold"})
    else:
        df["gold"] = ""
    img_cols = [c for c in df.columns if any(k in str(c).lower() for k in
                    ["image","images","image_url","figure","figures","pic","picture","img"])]
    df["image_refs"] = df.apply(lambda row: collect_image_refs(row, img_cols), axis=1)
    df["question"] = df["question"].astype(str)
    df["gold"] = df["gold"].astype(str)

    return df[["question","gold","image_refs"]].reset_index(drop=True)

def collect_image_refs(row, img_cols):
    refs = []
    for c in img_cols:
        v = row.get(c, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for x in v:
                s = str(x).strip()
                if s and s not in refs:
                    refs.append(s)
            continue
        if isinstance(v, dict):
            for x in v.values():
                s = str(x).strip()
                if s and s not in refs:
                    refs.append(s)
            continue
        s = str(v).strip()
        if not s:
            continue
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    for x in data:
                        sx = str(x).strip()
                        if sx and sx not in refs:
                            refs.append(sx)
                    continue
                if isinstance(data, dict):
                    for x in data.values():
                        sx = str(x).strip()
                        if sx and sx not in refs:
                            refs.append(sx)
                    continue
            except Exception:
                pass
        if "||" in s:
            for x in s.split("||"):
                sx = x.strip()
                if sx and sx not in refs:
                    refs.append(sx)
        else:
            if s not in refs:
                refs.append(s)
    return refs

def guess_mime(path_or_name: str) -> str:
    mt, _ = mimetypes.guess_type(path_or_name)
    return mt or "image/png"

class ZipImageIndex:
    def __init__(self, zip_root: str):
        self.zip_root = zip_root
        self.by_fullpath = {} 
        self.by_basename = defaultdict(list)  
        self._bytes_cache = OrderedDict()
        self._cache_cap = 64

    def _scan(self):
        for root, _dirs, files in os.walk(self.zip_root):
            for fn in files:
                if fn.lower().endswith(".zip"):
                    zp = os.path.join(root, fn)
                    try:
                        with zipfile.ZipFile(zp, "r") as zf:
                            for inner in zf.namelist():
                                key_full = inner.lower().lstrip("./\\")
                                self.by_fullpath[key_full] = (zp, inner)
                                base = os.path.basename(inner).lower()
                                self.by_basename[base].append((zp, inner))
                    except Exception:
                        continue

    @staticmethod
    def _norm_ref(ref: str) -> str:
        return str(ref or "").replace("\\", "/").lstrip("./").lower()

    def _cache_put(self, key, value):
        self._bytes_cache[key] = value
        self._bytes_cache.move_to_end(key)
        if len(self._bytes_cache) > self._cache_cap:
            self._bytes_cache.popitem(last=False)

    def _cache_get(self, key):
        v = self._bytes_cache.get(key)
        if v is not None:
            self._bytes_cache.move_to_end(key)
        return v

    def locate(self, ref: str):
        if not ref:
            return None
        refn = self._norm_ref(ref)
        if refn in self.by_fullpath:
            return self.by_fullpath[refn]
        hits = [(zp, inner) for key, (zp, inner) in self.by_fullpath.items()
                if key.endswith(refn)]
        if hits:
            return sorted(hits, key=lambda x: len(x[1]), reverse=True)[0]
        base = os.path.basename(refn)
        cand = self.by_basename.get(base, [])
        if cand:
            return cand[0]
        return None

    def read_bytes(self, ref: str):
        loc = self.locate(ref)
        if not loc:
            return None
        zp, inner = loc
        cache_key = f"{zp}::{inner}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                b = zf.read(inner)
            self._cache_put(cache_key, b)
            return b
        except Exception:
            return None

def build_user_content(question: str, img_payloads: list[tuple[str, bytes]]) -> list:
    parts = [{"type": "text", "text": (
        "You are a careful visual question answering assistant.\n"
        "Look at the image(s) first, then answer the question.\n"
        "Reply with ONLY the short final answer (no explanation, no extra words)."
    )}]
    for name, b in img_payloads:
        mime = guess_mime(name)
        b64 = base64.b64encode(b).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        parts.append({"type": "image_url", "image_url": {"url": data_uri}})
    parts.append({"type": "text", "text": f"Question:\n{question}\n\nAnswer:"})
    return parts

def ask_openai_mm(client: OpenAI, model: str, question: str, images: list[tuple[str, bytes]]) -> str:
    msgs = [
        {"role": "system", "content": "You are a concise, reliable visual QA assistant."},
        {"role": "user", "content": build_user_content(question, images)}
    ]
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                max_tokens=MAX_OUTPUT_TOKENS
            )
            return (resp.choices[0].message.content or "").strip()
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE ** attempt)
                continue
            return f"[Error: {e}]"
        except Exception as e:
            return f"[Error: {e}]"
    return f"[Error: {last_err}]" if last_err else "[Error: Unknown]"

def eval_bixbench_mm(jsonl_path: str, zip_root: str, outdir: str, model: str, limit: int | None):
    os.makedirs(outdir, exist_ok=True)
    df = load_bixbench(jsonl_path)
    if limit and limit > 0:
        df = df.head(limit).copy()

    out_csv = os.path.join(outdir, f"BixBench_{os.path.basename(model)}_multimodal.csv")
    done = 0
    if os.path.exists(out_csv):
        try:
            done = len(pd.read_csv(out_csv))
        except Exception:
            done = 0

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "Your key"))
    if not client.api_key:
        raise SystemExit("❌ Please set OPENAI_API_KEY")

    print(f"[INFO] Scanning ZIPs under: {zip_root}")
    img_index = ZipImageIndex(zip_root)

    n = len(df)
    for beg in range(done, n, CHUNK_SIZE):
        end = min(beg + CHUNK_SIZE, n)
        chunk = df.iloc[beg:end].copy()

        preds, scores, errors, found_imgs = [], [], [], []

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"BixBench[{beg}:{end})"):
            q = row["question"]
            gold = row.get("gold", "")
            refs: list[str] = row.get("image_refs", []) or []
            imgs = []
            found_names = []
            for r in refs:
                if len(imgs) >= MAX_IMAGES_PER_SAMPLE:
                    break
                b = img_index.read_bytes(r)
                if b:
                    imgs.append( (os.path.basename(r), b) )
                    found_names.append(r)
            found_imgs.append("|".join(found_names))
            ans = ask_openai_mm(client, model, q, imgs)

            if ans.startswith("[Error:"):
                pred = ""
                score = float("nan") if not str(gold).strip() else 0.0
                errors.append(ans)
            else:
                pred = ans.split("\n")[0].strip()
                if not str(gold).strip():
                    score = float("nan")
                else:
                    score = 1.0 if norm(pred) == norm(gold) else 0.0
                errors.append("")

            preds.append(pred)
            scores.append(score)
            gc.collect()

        chunk["prediction"] = preds
        chunk["correct"] = scores
        chunk["error"] = errors
        chunk["found_images"] = found_imgs

        cols = ["question","gold","image_refs","found_images","prediction","correct","error"]
        chunk[cols].to_csv(out_csv, mode="a" if beg else "w", index=False, header=not beg)

    full = pd.read_csv(out_csv)
    full["correct"] = pd.to_numeric(full["correct"], errors="coerce")
    acc = float(full["correct"].mean(skipna=True)) if len(full) else 0.0
    print(f"[RESULT] ACC={acc*100:.2f}% → {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to BixBench.jsonl")
    ap.add_argument("--zip_root", type=str, required=True, help="Directory that contains image ZIPs")
    ap.add_argument("--outdir", type=str, default=r".\results_bix_mm")
    ap.add_argument("--model", type=str, default="gpt-4o", help="OpenAI multimodal model id (e.g., gpt-4o, gpt-4o-mini, gpt-5-mini)")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate first N rows only (0 = all)")
    args = ap.parse_args()

    eval_bixbench_mm(
        jsonl_path=args.jsonl,
        zip_root=args.zip_root,
        outdir=args.outdir,
        model=args.model,
        limit=(args.limit if args.limit > 0 else None)
    )

if __name__ == "__main__":
    main()
