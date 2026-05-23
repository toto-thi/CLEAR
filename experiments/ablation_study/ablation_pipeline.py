import sys
import json
import argparse
import pickle as pkl
from pathlib import Path
from typing import TypedDict, Dict, List, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from experiments.llm_setup import LLMManager
from experiments.eval_runner import BatchEvaluator

class DermState(TypedDict):
    """LangGraph state definition for the ablation workflow (No Retrieval)."""
    patient_data: Dict[str, Any]
    image_path: str
    lab_report: Dict[str, Any]
    triage_result: Dict[str, Any]
    diagnose_result: Dict[str, Any]
    trace: List[Dict[str, Any]]

def get_prompt(name: str) -> str:
    """Helper to load agent prompts from the 'agents/' directory."""
    prompt_path = project_root / "agents" / f"{name.upper()}.md"
    return prompt_path.read_text()

# Custom Diagnosis Prompt for Ablation (Removes retrieval dependency)
DX_ABLATION_PROMPT = """# Diagnosis Agent Prompt (Ablation)

You are a specialist AI Dermoscopist. Your task is to propose a primary diagnosis and a differential by building the strongest possible argument based on visual evidence.

### Allowed Diagnoses by Family:
* **Melanocytic:** Nevus, Melanoma
* **Keratinocytic:** Basal cell carcinoma, Squamous cell carcinoma, Pigmented benign keratosis, Actinic keratosis
* **Fibrohistiocytic:** Dermatofibroma

### Operational Workflow:
1. Provide a `query_summary` based on your direct visual inspection of the lesion image.
2. Acknowledge that diseases can present atypically or mimic other conditions.
3. In your final synthesis, perform a comparative analysis. Directly weigh the positive evidence for your chosen diagnosis against the evidence for your main differential diagnosis to justify your conclusion.

### Required Output Format:
```json
{
  "diagnosis": "<final diagnosis>",
  "confidence": "<Low | Medium | High>",
  "differential_diagnosis": "<secondary diagnosis from allowed list>",
  "reasoning": {
    "query_summary": "<Visual description...>",
    "synthesis": "<Clear conclusion justifying the choice...>"
  }
}
```"""

def create_ablation_graph(llm_manager: LLMManager):
    """Constructs the LangGraph workflow for ablation (Lab -> Triage -> Diagnosis)."""
    models = llm_manager.models
    
    # Load core prompts from agents/
    prompts = {
        "lab": get_prompt('lab'),
        "triage": get_prompt('triage'),
        "diagnosis": DX_ABLATION_PROMPT
    }

    def lab_technician(state: DermState):
        image_path = state.get("image_path", "")
        image_data = llm_manager.encode_image(image_path)
        patient_data = state.get("patient_data", {})
        
        user_message = [
            { "type": "text", "text": "Process the following case and generate the initial report.\n" },
            { "type": "text", "text": f"Patient Data:\n {json.dumps(patient_data)}\n" },
            { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_data}" } }
        ]
        
        lab_report = llm_manager.invoke_llm(models.lab_tech, prompts["lab"], user_message)  
        llm_manager.add_trace(state, agent="lab_technician", role="assistant", payload=lab_report)
        return {"lab_report": lab_report}

    def triage_node(state: DermState):
        lab_report = state.get("lab_report", {})
        image_data = llm_manager.encode_image(state.get("image_path", ""))
        
        user_message = [
            { "type": "text", "text": "Based on the following report AND the provided image, perform triage and classify the disease family.\n" },
            { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_data}" } },
            { "type": "text", "text": f"Lab Report:\n{json.dumps(lab_report, ensure_ascii=False)}" },
        ]

        triage_result = llm_manager.invoke_llm(models.triage, prompts["triage"], user_message)
        llm_manager.add_trace(state, agent="triage", role="assistant", payload=triage_result)
        return {"triage_result": triage_result}

    def diagnosis_node(state: DermState):
        lab_report = state.get("lab_report", {})
        triage_result = state.get("triage_result", {})
        image_path = state.get("image_path", "")
        image_data = llm_manager.encode_image(image_path)
        
        user_message = [
            { "type": "text", "text": "Provide a final diagnosis by synthesizing all available evidence.\n" },
            { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_data}" } },
            { "type": "text", "text": f"Lab Report:\n{json.dumps(lab_report, ensure_ascii=False)}" },
            { "type": "text", "text": f"Triage Result:\n{json.dumps(triage_result, ensure_ascii=False)}" },
        ]

        dx_result = llm_manager.invoke_llm(models.diagnose, prompts["diagnosis"], user_message)
        llm_manager.add_trace(state, agent="diagnose", role="assistant", payload=dx_result)

        return {"diagnose_result": dx_result}

    # Build Graph
    builder = StateGraph(DermState)
    builder.add_node("lab_technician", lab_technician)
    builder.add_node("triage", triage_node)
    builder.add_node("diagnosis", diagnosis_node)

    builder.set_entry_point("lab_technician")
    builder.add_edge("lab_technician", "triage")
    builder.add_edge("triage", "diagnosis")
    builder.add_edge("diagnosis", END)

    return builder.compile(checkpointer=InMemorySaver())

def main():
    parser = argparse.ArgumentParser(description="Dermatological Ablation Pipeline (No VKB) CLI")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"], help="LLM provider")
    parser.add_argument("--output", type=str, help="Specific output filename in 'results/ablation/' directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress")
    args = parser.parse_args()

    # Setup Environment
    results_dir = project_root / "results" / "ablation"
    results_dir.mkdir(exist_ok=True, parents=True)
    out_file = args.output or f"{args.provider}.jsonl"

    # Load Dataset
    dataset_path = project_root / "dataset" / "300_test_set.pkl"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "rb") as f:
        test_dataset = pkl.load(f)
    
    # Fix image paths
    for item in test_dataset:
        original_path = Path(item["image_path"])
        item["image_path"] = str(project_root / "dataset" / "test" / original_path.name)

    # Initialize components
    print(f"Initializing Ablation Pipeline using {args.provider.upper()}...")
    llm_manager = LLMManager(provider=args.provider)
    app = create_ablation_graph(llm_manager)

    # Run batch evaluation
    evaluator = BatchEvaluator(
        app=app,
        out_dir=str(results_dir),
        out_file=out_file,
        resume=args.resume
    )
    
    evaluator.run(test_dataset)

if __name__ == "__main__":
    main()
