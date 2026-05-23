import re
import itertools
import numpy as np
from typing import Set, Tuple, List, Dict, Any
from sklearn.metrics import confusion_matrix
from .constants import FAMILY_FROM_DX

def specificity_multiclass(y_true, y_pred) -> Dict[str, Any]:
    """
    Compute specificity for each class in a multi-class setting.
    """
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_classes = cm.shape[0]
    specificities = []

    # Support: count of true instances per class
    support = cm.sum(axis=1)

    for i in range(n_classes):
        # True Negatives for class i: sum of all cells not in row i or column i
        tn = np.sum(cm[np.arange(n_classes) != i][:, np.arange(n_classes) != i])
        
        # False Positives for class i: sum of column i, excluding diagonal (TP)
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        # Specificity for class i
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity_i)
    
    specificities = np.array(specificities)
    
    results = {
        'per_class': {labels[i]: spec for i, spec in enumerate(specificities)},
        'macro_avg': np.mean(specificities),
        'weighted_avg': np.average(specificities, weights=support) if np.sum(support) > 0 else 0.0
    }
    
    return results

def sent_tokenize(s: str) -> list:
    """
    Simple sentence split (no external deps).
    """
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    parts = re.split(r'(?<=[\.!?])\s+', s)
    return [p.strip() for p in parts if p.strip()]

def collect_observed_tokens(case: Dict[str, Any]) -> Set[str]:
    seen: Set[str] = set()
    # From lab report features
    lab = case.get("lab_report", {})
    vf = lab.get("visual_summary", {})
    
    # Add visual summary tokens
    for k in ["symmetry","border_characteristics"]:
        v = vf.get(k)
        if isinstance(v, str):
            seen.update(re.findall(r"[a-zA-Z\-]+", v.lower()))
        elif isinstance(v, list):
            for item in v:
                seen.update(re.findall(r"[a-zA-Z\-]+", str(item).lower()))
    
    # Dermoscopic features
    for feat in lab.get("dermoscopic_features", []):
        fn = feat.get("feature_name","")
        fd = feat.get("description","")
        seen.update(re.findall(r"[a-zA-Z\-]+", (fn+" "+fd).lower()))
    
    # Diagnose query summary (direct image read)
    diag_qs = case.get("diagnose_result", {}).get("reasoning", {}).get("query_summary", "")
    seen.update(re.findall(r"[a-zA-Z\-]+", diag_qs.lower()))
    
    return seen

def contains_lexicon_term(sentence: str, lexicon: Set[str]) -> bool:
    s = sentence.lower()
    return any(term in s for term in lexicon)

def is_quantified_or_spatial(sentence: str) -> bool:
    """
    Check for numbers, percentages, or spatial terms like 'central', 'peripheral', 'border', 'asymmetry'.
    """
    return bool(re.search(r'\d|percent|%|central|peripheral|upper|lower|left|right|border|symmetr|asymmetr', sentence.lower()))

def rationale_specificity(texts: list, lexicon: Set[str]) -> float:
    sents = list(itertools.chain.from_iterable(sent_tokenize(t) for t in texts))
    if not sents:
        return float('nan')
    
    specific = 0
    for s in sents:
        if contains_lexicon_term(s, lexicon) or is_quantified_or_spatial(s):
            specific += 1
    return specific / len(sents)

def grounded_and_unsupported_counts(case: Dict[str, Any], lexicon: Set[str]) -> Tuple[int, int, int]:
    """
    Returns: grounded_count, unsupported_count, total_claims
    We treat each sentence containing any lexicon keyword as a "claim".
    It's grounded if any lexicon token in the sentence also appears among observed tokens from lab/query.
    Otherwise, it's unsupported.
    """
    texts = []
    diag = case.get("diagnose_result", {})
    diag_r = diag.get("reasoning", {})
    texts.append(diag_r.get("query_summary", ""))
    for item in diag_r.get("comparative_analysis", []) or []:
        texts.append(item.get("justification", ""))
    texts.append(diag_r.get("synthesis", ""))
    
    triage_r = case.get("triage_result", {}).get("reasoning", "")
    texts.append(triage_r or "")
    
    final_r = case.get("final_report", {}).get("reasoning", "")
    texts.append(final_r or "")

    observed = collect_observed_tokens(case)
    sents = list(itertools.chain.from_iterable(sent_tokenize(t) for t in texts))
    
    total_claims = 0
    grounded = 0
    unsupported = 0
    
    for s in sents:
        s_low = s.lower()
        if contains_lexicon_term(s_low, lexicon):
            total_claims += 1
            # extract the lexicon terms present in sentence
            terms_present = [t for t in lexicon if t in s_low]
            grounded_flag = False
            for t in terms_present:
                for w in re.findall(r"[a-zA-Z\-]+", t):
                    if w in observed:
                        grounded_flag = True
                        break
                if grounded_flag:
                    break
            if grounded_flag:
                grounded += 1
            else:
                unsupported += 1
    return grounded, unsupported, total_claims

def family_of(dx: str) -> str:
    """
    Map a disease name to its family.
    """
    return FAMILY_FROM_DX.get(dx, "Unknown")

def collect_relevant_terms(case: Dict[str, Any]) -> Set[str]:
    """
    Collect relevant terms (from lab report, triage result, diagnose result, critique report, etc.)
    that are used for reasoning in later reports.
    """
    terms = set()
    
    # Collecting terms from the lab report dermoscopic features
    lab_report = case.get("lab_report", {})
    dermoscopic_features = lab_report.get("dermoscopic_features", [])
    for feature in dermoscopic_features:
        terms.add(feature.get("feature_name", "").lower())
        terms.add(feature.get("description", "").lower())
    
    # Collecting terms from the visual summary
    visual_summary = lab_report.get("visual_summary", {})
    for key, value in visual_summary.items():
        if isinstance(value, str):
            terms.update(re.findall(r"[a-zA-Z\-]+", value.lower()))
        elif isinstance(value, list):
            for item in value:
                terms.update(re.findall(r"[a-zA-Z\-]+", str(item).lower()))
    
    # Collecting terms from diagnose result (query summary, comparative analysis)
    diag_result = case.get("diagnose_result", {}).get("reasoning", {})
    if diag_result.get("query_summary"):
        terms.update(re.findall(r"[a-zA-Z\-]+", diag_result["query_summary"].lower()))
    for item in diag_result.get("comparative_analysis", []):
        terms.update(re.findall(r"[a-zA-Z\-]+", item.get("justification", "").lower()))
    
    # Collecting terms from triage result reasoning
    triage_result = case.get("triage_result", {}).get("reasoning", "")
    terms.update(re.findall(r"[a-zA-Z\-]+", triage_result.lower()))
    
    # Collecting terms from critique report (counterarguments)
    critique_report = case.get("critique_report", {}).get("critique_assessment", {}).get("counterargument", "")
    terms.update(re.findall(r"[a-zA-Z\-]+", critique_report.lower()))
    
    # Collecting terms from final report reasoning
    final_report = case.get("final_report", {}).get("reasoning", "")
    terms.update(re.findall(r"[a-zA-Z\-]+", final_report.lower()))
    
    return terms

def evidence_citation_rate(case: Dict[str, Any]) -> float:
    """
    Calculate the citation rate based on whether reasoning uses terms from lab report, triage result, 
    diagnosis result, critique report, or final report.
    """
    relevant_terms = collect_relevant_terms(case)
    texts = []
    
    diag_result = case.get("diagnose_result", {}).get("reasoning", {})
    texts.append(diag_result.get("synthesis", ""))
    
    triage_result = case.get("triage_result", {}).get("reasoning", "")
    texts.append(triage_result)
    
    final_report = case.get("final_report", {}).get("reasoning", "")
    texts.append(final_report)
    
    critique_report = case.get("critique_report", {}).get("critique_assessment", {}).get("counterargument", "")
    texts.append(critique_report)
    
    sents = list(itertools.chain.from_iterable(sent_tokenize(t) for t in texts))
    
    cited_sentences = 0
    valid_citations = 0
    
    for sentence in sents:
        cited = False
        for term in relevant_terms:
            if term in sentence.lower():
                cited = True
                break
        
        if cited:
            cited_sentences += 1
            if all(term in sentence.lower() for term in relevant_terms if term in sentence.lower()):
                valid_citations += 1
    
    return valid_citations / cited_sentences if cited_sentences > 0 else float('nan')

def recall_at_k(y_true: List[str], retrieved_cases_list: List[List[Dict[str, Any]]]) -> float:
    """
    Calculates the Recall@K score.
    """
    scores = []
    for gt, top_k in zip(y_true, retrieved_cases_list):
        if not top_k:
            scores.append(0.0)
            continue
        # Extract diagnoses from retrieved cases (sorted by similarity score)
        retrieved_diagnoses = [entry.get('diagnosis') for entry in sorted(top_k, key=lambda x: x.get('sim_score', 0), reverse=True)]
        scores.append(1.0 if gt in retrieved_diagnoses else 0.0)
    return np.mean(scores)

def diversity_at_k(retrieved_cases_list: List[List[Dict[str, Any]]], k: int = 3) -> float:
    """
    Calculates the Diversity@K score (proportion of unique diagnoses among K retrieved cases).
    """
    scores = []
    for top_k in retrieved_cases_list:
        if not top_k:
            scores.append(0.0)
            continue
        unique_diagnoses = len(set([entry.get('diagnosis') for entry in top_k]))
        scores.append(unique_diagnoses / k)
    return np.mean(scores)

def contradiction_flags(case: Dict[str, Any]) -> Dict[str, bool]:
    flags = {}
    
    # final diagnosis family vs triage family
    final_dx = case.get("final_report", {}).get("final_diagnosis")
    triage_family = case.get("triage_result", {}).get("disease_family", "")
    flags["final_vs_triage_family"] = (family_of(final_dx) != triage_family) if final_dx and triage_family else False
    
    # border contradiction (visual vs reasoning)
    vis_border = case.get("lab_report", {}).get("visual_summary", {}).get("border_characteristics", "")
    rationale_text = " ".join([
        case.get("diagnose_result", {}).get("reasoning", {}).get("query_summary", ""),
        case.get("final_report", {}).get("reasoning", ""),
        case.get("triage_result", {}).get("reasoning", "")
    ]).lower()
    
    says_well_defined = "well-defined" in rationale_text or "well defined" in rationale_text
    has_ill_irreg = "ill-defined" in vis_border.lower() or "irregular" in vis_border.lower()
    flags["border_contradiction"] = bool(says_well_defined and has_ill_irreg)

    # color contradiction (visual vs reasoning)
    vis_colors = case.get("lab_report", {}).get("visual_summary", {}).get("colors", [])
    flags["color_contradiction"] = not any(color in rationale_text for color in vis_colors)

    # symmetry contradiction (visual vs reasoning)
    vis_symmetry = case.get("lab_report", {}).get("visual_summary", {}).get("symmetry", "")
    says_asymmetrical = "asymmetrical" in rationale_text
    flags["symmetry_contradiction"] = vis_symmetry.lower() != "asymmetrical" and says_asymmetrical
    
    return flags

def evaluate_cases(cases: List[Dict[str, Any]], labels: List[str] = None, lexicon: Set[str] = None) -> Dict[str, Any]:
    """
    High-level evaluation function to process a list of cases and compute aggregate metrics.
    """
    if lexicon is None:
        from .constants import DEFAULT_LEXICON
        lexicon = DEFAULT_LEXICON
        
    y_true = [c.get("ground_truth") for c in cases if c.get("ground_truth")]

    spec_vals = []
    grounded_cnt = 0
    unsupported_cnt = 0
    total_claims = 0
    citations = []
    contradictions = []

    for c in cases:
        # Rationale specificity
        diag_r = (c.get("diagnose_result", {}).get("reasoning", {}) or {})
        texts = [
            diag_r.get("query_summary",""),
            " ".join(j.get("justification","") for j in (diag_r.get("comparative_analysis") or [])),
            diag_r.get("synthesis",""),
            c.get("triage_result", {}).get("reasoning","") or "",
            c.get("final_report", {}).get("reasoning","") or "",
        ]
        spec_vals.append(rationale_specificity(texts, lexicon=lexicon))

        # Grounded & unsupported
        g, u, t = grounded_and_unsupported_counts(c, lexicon=lexicon)
        grounded_cnt += g
        unsupported_cnt += u
        total_claims += t

        # Citation rate
        citations.append(evidence_citation_rate(c))

        # Contradictions
        contradictions.append(any(contradiction_flags(c).values()))

    grounded_rate = grounded_cnt/total_claims if total_claims > 0 else float('nan')
    unsupported_rate = unsupported_cnt/total_claims if total_claims > 0 else float('nan')
    citation_rate = np.nanmean(citations) if citations else float('nan')
    contradiction_rate = np.mean(contradictions) if contradictions else float('nan')

    return {
        "overall": {
            "N": len(y_true),
            "Evidence Citation Rate": citation_rate,
            "Grounded-evidence Rate": grounded_rate,
            "Unsupported-claim Rate": unsupported_rate,
            "Contradiction Rate": contradiction_rate
        }
    }
