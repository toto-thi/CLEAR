import json
import re
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from .constants import LABEL_MAPPING

def clean_raw_json(raw_str: str) -> str:
    """
    Fixes common JSON formatting issues in LLM outputs (e.g., control characters, 
    incorrectly escaped newlines, missing braces).
    """
    if not raw_str:
        return "{}"
        
    cleaned = ''.join(
        c if ord(c) >= 32 and c != '\x7f' else ' '
        for c in raw_str
    )
    
    # Standardize newlines and carriage returns
    cleaned = re.sub(r'(?<!\\)(?:\\\\)*\n', '\\\\n', cleaned)
    cleaned = re.sub(r'(?<!\\)(?:\\\\)*\r', '\\\\r', cleaned)

    # Ensure it starts with {
    stripped = cleaned.strip()
    if not stripped.startswith('{'):
        pos = cleaned.find('{')
        if pos != -1:
            cleaned = cleaned[pos:]
        else:
            cleaned = '{' + cleaned

    # Ensure it ends with }
    if not cleaned.strip().endswith('}'):
        pos = cleaned.rfind('}')
        if pos != -1:
            cleaned = cleaned[:pos+1]
        else:
            cleaned = cleaned + '}'

    return cleaned

def normalize_label(label: Any) -> Any:
    """
    Standardizes diagnosis labels using LABEL_MAPPING.
    """
    if isinstance(label, str):
        return LABEL_MAPPING.get(label.strip(), label.strip())
    return label

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file into a list of dictionaries.
    """
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))
    return cases

def parse_payload(payload: Any) -> Dict[str, Any]:
    """
    Helper to parse a payload that might contain a '_raw' JSON string.
    """
    if not payload:
        return {}
    if isinstance(payload, dict):
        if '_raw' in payload and payload['_raw']:
            try:
                # Some _raw might have markdown code blocks
                raw_str = payload['_raw']
                if '```json' in raw_str:
                    raw_str = re.search(r'```json\s*(.*?)\s*```', raw_str, re.DOTALL).group(1)
                
                cleaned = clean_raw_json(raw_str)
                return json.loads(cleaned)
            except (json.JSONDecodeError, TypeError, AttributeError):
                return {}
        return payload
    return {}

def extract_evaluation_data(df: pd.DataFrame, target_agent: str = "synthesizer") -> pd.DataFrame:
    """
    Extracts structured evaluation data from a results dataframe containing traces.
    
    Args:
        df: DataFrame with 'trace', 'ground_truth', 'image_path', and optionally 'retrieved_cases'.
        target_agent: The agent from which to extract the final diagnosis (e.g., 'synthesizer' or 'diagnose').
    """
    processed_data = []
    
    for index, item in df.iterrows():
        trace = item.get("trace", [])
        ground_truth = normalize_label(item.get("ground_truth"))
        image_path_full = item.get("image_path", "")
        image_id = Path(image_path_full).stem if image_path_full else f"unknown_{index}"
        
        # Extract retrieved cases if present
        retrieved_cases = []
        if "retrieved_cases" in item and isinstance(item["retrieved_cases"], list):
            for rc in item["retrieved_cases"]:
                retrieved_cases.append({
                    "image_path": Path(rc.get("image_path", "")).stem,
                    "age": rc.get("age"),
                    "sex": rc.get("sex"),
                    "diagnosis": normalize_label(rc.get("diagnosis")),
                    "lesion_location": rc.get("anatom_site"),
                    "melanocytic": rc.get("melanocytic"),
                    "sim_score": rc.get("score")
                })

        # Find the target agent's payload in the trace
        agent_payload_entry = next((t for t in reversed(trace) if t.get("agent") == target_agent), None)
        
        if not agent_payload_entry:
            # Fallback for ablation or if target not found
            processed_data.append({
                "image_path": image_id,
                "diagnosis": None,
                "reasoning": None,
                "confidence": None,
                "differential": None,
                "ground_truth": ground_truth,
                "retrieved_cases": retrieved_cases if retrieved_cases else None
            })
            continue

        data = parse_payload(agent_payload_entry.get("payload"))
        
        # Field mapping (handles slight variations in schemas)
        diagnosis = data.get("final_diagnosis") or data.get("diagnosis")
        confidence = data.get("confidence")
        differential = data.get("differential_diagnosis")
        
        reasoning = data.get("reasoning", "")
        if isinstance(reasoning, dict):
            reasoning = reasoning.get("synthesis", "")

        processed_data.append({
            "image_path": image_id,
            "diagnosis": normalize_label(diagnosis),
            "reasoning": reasoning,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "differential": differential,
            "retrieved_cases": retrieved_cases if retrieved_cases else None
        })

    return pd.DataFrame(processed_data)

def align_evaluation_data(df_ini: pd.DataFrame, df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns initial and final diagnosis dataframes on 'image_path'.
    Returns a merged dataframe containing only cases present in both.
    """
    # Merge on image_path and ground_truth
    merged = pd.merge(
        df_ini[['image_path', 'diagnosis', 'ground_truth']], 
        df_final, 
        on=['image_path', 'ground_truth'], 
        suffixes=('_ini', '_final')
       )
    
    return merged

