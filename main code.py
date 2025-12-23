import os
import re
import json
import time
import argparse
import requests
import pandas as pd

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "[OLLAMA_HOST_PLACEHOLDER]")
OLLAMA_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "[MODEL_NAME_PLACEHOLDER]")
REQUEST_TIMEOUT = 600
SLEEP_BETWEEN_CALLS_SEC = 0.5

OLLAMA_OPTIONS = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "num_predict": 2048,
    "num_ctx": 8192
}

INPUT_CSV_DEFAULT = "[INPUT_PATH_PLACEHOLDER]"
OUTPUT_CSV_DEFAULT = "[OUTPUT_PATH_PLACEHOLDER]"

START_ID = 0
END_ID = -1

PROMPT_TEMPLATE = """
YOUR ROLE:
You are an expert in biogerontology and computational health analytics, specializing in estimating biological age from multidimensional clinical data. You integrate biomarkers, physiological metrics, and health history to infer both systemic and organ-level aging patterns. You are skilled at synthesizing diverse health information to generate biologically meaningful estimates of aging rates across the human body.

BACKGROUND KNOWLEDGE:
One indicator in aging assessment is called Biological Age, which represents the cumulative physiological wear and tear of an individual. It may deviate from chronological age, or may also equal the chronological age. A biological age higher than chronological age implies accelerated aging or reduced systemic resilience, whereas a lower biological age indicates relative youthfulness and healthier system integrity. Different organ systems often age at distinct rates. Therefore, accurate biological age estimation requires holistic interpretation across multiple indicators, including blood biochemistry, imaging, lifestyle habits, and clinical records. Additionally, aging progresses unevenly across bodily systems (for instance, cardiovascular aging might differ from neurocognitive aging), necessitating separate evaluations for overall and system-specific ages.

YOUR TASK:
Using the provided health data, infer the individual’s overall biological age and system-specific biological ages for each major physiological domain. Before presenting the numerical estimates, briefly describe your reasoning process for each system, focusing on how specific indicators influence the inferred aging of that domain. Your output must follow strict JSON format with no extra commentary. Follow the schema below exactly:

```json
{
“inference process 1”: string, // Explain how you inferred the overall biological age.
“overall biological age”: int,
“inference process 2”: string, // Explain how you inferred the cardiovascular system age.
“cardiovascular system age”: int,
“inference process 3”: string, // Explain how you inferred the metabolic/endocrine system age.
“metabolic/endocrine system age”: int,
“inference process 4”: string, // Explain how you inferred the hepatic system age.
“hepatic system age”: int,
“inference process 5”: string, // Explain how you inferred the renal system age.
“renal system age”: int,
“inference process 6”: string, // Explain how you inferred the immune/inflammatory system age.
“immune/inflammatory system age”: int,
“inference process 7”: string, // Explain how you inferred the respiratory system age.
“respiratory system age”: int,
“inference process 8”: string, // Explain how you inferred the neurocognitive system age.
“neurocognitive system age”: int,
“inference process 9”: string, // Explain how you inferred the oral-sensory system age.
“oral-sensory system age”: int
}

You will be provided with the individual's complete health data as follows:
{health_record}

Based on the health data above, now output the JSON:

"""

OUTPUT_FIELDS = [
    "inference process 1", "overall biological age",
    "inference process 2", "cardiovascular system age",
    "inference process 3", "metabolic/endocrine system age",
    "inference process 4", "hepatic system age",
    "inference process 5", "renal system age",
    "inference process 6", "immune/inflammatory system age",
    "inference process 7", "respiratory system age",
    "inference process 8", "neurocognitive system age",
    "inference process 9", "oral-sensory system age",
]

ALL_OUTPUT_COLUMNS = ["person_id"] + OUTPUT_FIELDS + ["prompt_sent", "raw_response", "error"]

def normalize_key(k):
    if not isinstance(k, str):
        return k
    k = k.strip().lower()
    k = re.sub(r"[\s–—\-]+", " ", k)
    k = k.replace("–", " ").replace("-", " ").strip()
    
    for field in OUTPUT_FIELDS:
        f_norm = re.sub(r"[\s–—\-]+", " ", field.lower())
        f_norm = f_norm.replace("–", " ").replace("-", " ").strip()
        if k == f_norm:
            return field
    return k

def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end+1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "options": OLLAMA_OPTIONS,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")
    except requests.RequestException as e:
        raise RuntimeError(f"HTTP error: {e}")

def process_one(pid, health_record):
    prompt = PROMPT_TEMPLATE.replace("{health_record}", str(health_record))
    raw = call_ollama(prompt)
    
    obj = extract_json(raw)
    if not obj:
        raise ValueError(f"Invalid JSON. Snippet: {repr(raw[:150])}")
    
    normalized_obj = {}
    for k, v in obj.items():
        normalized_key = normalize_key(k)
        normalized_obj[normalized_key] = v
    
    missing = [f for f in OUTPUT_FIELDS if f not in normalized_obj]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Got keys: {list(normalized_obj.keys())}")
    
    for f in OUTPUT_FIELDS:
        if "age" in f:
            val = normalized_obj[f]
            if not isinstance(val, int):
                try:
                    normalized_obj[f] = int(float(val))
                except:
                    raise ValueError(f"Age field '{f}' is not an integer: {val}")
    
    row = {"person_id": pid}
    for f in OUTPUT_FIELDS:
        row[f] = normalized_obj[f]
    row["prompt_sent"] = prompt  
    row["raw_response"] = raw   
    row["error"] = None       
    return row

def save_checkpoint(rows, out_csv):
    df_out = pd.DataFrame(rows, columns=ALL_OUTPUT_COLUMNS)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} rows to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_CSV_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_CSV_DEFAULT)
    parser.add_argument("--limit", type=int, default=END_ID - START_ID + 1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args, unknown = parser.parse_known_args()

    df = pd.read_csv(args.input)
    if "person_id" not in df.columns or "health_record" not in df.columns:
        raise ValueError("Input CSV must have 'person_id' and 'health_record'.")

    if START_ID is not None or END_ID is not None:
        print(f"Filtering records from ID {START_ID} to ID {END_ID}")
        
        if "person_id" not in df.columns:
            raise ValueError("CSV file must contain 'person_id' column for ID filtering")
        
        if START_ID is not None and END_ID != -1:
            filtered_df = df[(df["person_id"] >= START_ID) & (df["person_id"] <= END_ID)]
        elif START_ID is not None and END_ID == -1:
            filtered_df = df[df["person_id"] >= START_ID]
        elif START_ID is None and END_ID != -1:
            filtered_df = df[df["person_id"] <= END_ID]
        else:
            filtered_df = df
        
        print(f"Found {len(filtered_df)} records in the specified ID range")
        
        filtered_df = filtered_df.sort_values("person_id")
        df = filtered_df
    else:
        total = min(args.limit, len(df))
        df = df.iloc[:total]

    if START_ID is None and END_ID is None:
        total = min(args.limit, len(df))
        df = df.iloc[:total]
    else:
        total = len(df)

    results = []

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=total, desc="Predicting", unit="sample")
    except ImportError:
        iterator = enumerate(df.iterrows())

    for i, (_, r) in enumerate(iterator, start=1):
        pid, record = r["person_id"], r["health_record"]
        try:
            row = process_one(pid, record)
        except Exception as e:
            prompt = PROMPT_TEMPLATE.replace("{health_record}", str(record))
            row = {
                "person_id": pid,
                **{f: None for f in OUTPUT_FIELDS},
                "prompt_sent": prompt,
                "raw_response": "",  
                "error": str(e)
            }
        results.append(row)
        if i % args.checkpoint_every == 0:
            save_checkpoint(results, args.output)
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    save_checkpoint(results, args.output)
    print("Done.")

if __name__ == "__main__":
    main()