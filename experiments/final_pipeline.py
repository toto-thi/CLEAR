import os
import sys
import json
import argparse
import pickle as pkl
from pathlib import Path
from typing import TypedDict, Dict, List, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.llm_setup import LLMManager
from experiments.eval_runner import BatchEvaluator
from experiments.retrieve_image import QdrantBiomedCLIPRetriever

# Constants for Knowledge Base
COLLECTION_NAME = "derm_vkb"
QDRANT_URL = "http://localhost:6333"

class DermState(TypedDict):
    """LangGraph state definition for the dermatological diagnosis workflow."""
    patient_data: Dict[str, Any]
    image_path: str
    lab_report: Dict[str, Any]
    triage_result: Dict[str, Any]
    diagnose_result: Dict[str, Any]
    critique_report: Dict[str, Any]
    final_report: Dict[str, Any] 
    retrieved_cases: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    ground_truth: str  # For evaluation tracking

def get_prompt(name: str) -> str:
    """Helper to load agent prompts from the 'agents/' directory."""
    prompt_path = project_root / "agents" / f"{name.upper()}.md"
    return prompt_path.read_text()

def create_diagnosis_graph(llm_manager: LLMManager):
    """Constructs and compiles the LangGraph workflow."""
    models = llm_manager.models
    
    # Pre-load prompts
    prompts = {
        "lab": get_prompt('lab'),
        "triage": get_prompt('triage'),
        "diagnosis": get_prompt('diagnosis'),
        "critique": get_prompt('critique'),
        "synthesizer": get_prompt('synthesizer'),
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
        is_melanocytic_flag = lab_report.get("patient_data", {}).get("is_melanocytic")
        triage_result = state.get("triage_result", {})
        image_path = state.get("image_path", "")
        image_data = llm_manager.encode_image(image_path)
        
        user_message = [
            { "type": "text", "text": "Provide a final diagnosis by synthesizing all evidence and comparing to reference cases.\n" },
            { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_data}" } },
            { "type": "text", "text": f"Lab Report:\n{json.dumps(lab_report, ensure_ascii=False)}" },
            { "type": "text", "text": f"Triage Result:\n{json.dumps(triage_result, ensure_ascii=False)}" },
        ]

        # VKB Retrieval
        retriever = QdrantBiomedCLIPRetriever(collection=COLLECTION_NAME, qdrant_url=QDRANT_URL)
        retrieved_cases = retriever.search(image_path=image_path, k=3)

        if retrieved_cases:
            user_message.append({"type": "text", "text": "\nReference Cases:"})
            for i, case in enumerate(retrieved_cases):
                case_image = llm_manager.encode_image(case.get("image_path", ""))
                user_message.append({
                    "type": "text",
                    "text": f"Reference Case #{i + 1}: Diagnosis: {case['diagnosis']}, age: {case['age']}, sex: {case['sex']}, lesion_site: {case['anatom_site']}\n"
                })
                user_message.append({ "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{case_image}" } }) 

        dx_result = llm_manager.invoke_llm(models.diagnose, prompts["diagnosis"], user_message)
        llm_manager.add_trace(state, agent="diagnose", role="assistant", payload=dx_result)

        return {"diagnose_result": dx_result, "retrieved_cases": retrieved_cases}

    def critique_node(state: DermState):
        lab_report = state.get("lab_report", {})
        triage_result = state.get("triage_result", {})
        diagnose_result = state.get("diagnose_result", {})
        
        user_message = f"Lab Report:\n{json.dumps(lab_report)}\nTriage Result:\n{json.dumps(triage_result)}\nProposed Diagnosis:\n{json.dumps(diagnose_result)}"

        critique_report = llm_manager.invoke_llm(models.critique, prompts["critique"], user_message)
        llm_manager.add_trace(state, agent="critique", role="assistant", payload=critique_report)
        return {"critique_report": critique_report}

    def synthesizer_node(state: DermState):
        diagnose_result = state.get("diagnose_result", {})
        critique_report = state.get("critique_report", {})
        
        user_message = f"Initial Proposal:\n{json.dumps(diagnose_result)}\nCritical Review:\n{json.dumps(critique_report)}"

        final_report = llm_manager.invoke_llm(models.synthesizer, prompts["synthesizer"], user_message)
        llm_manager.add_trace(state, agent="synthesizer", role="assistant", payload=final_report)
        return {"final_report": final_report}

    # Build Graph
    builder = StateGraph(DermState)
    builder.add_node("lab_technician", lab_technician)
    builder.add_node("triage", triage_node)
    builder.add_node("diagnosis", diagnosis_node)
    builder.add_node("critique", critique_node) 
    builder.add_node("synthesizer", synthesizer_node) 

    builder.set_entry_point("lab_technician")
    builder.add_edge("lab_technician", "triage")
    builder.add_edge("triage", "diagnosis")
    builder.add_edge("diagnosis", "critique")
    builder.add_edge("critique", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=InMemorySaver())

def main():
    parser = argparse.ArgumentParser(description="Dermatological Diagnosis Multi-Agent Pipeline CLI")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"], help="LLM provider (default: openai)")
    parser.add_argument("--output", type=str, help="Specific output filename in 'results/' directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress in the output file")
    args = parser.parse_args()

    # Setup Environment
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = args.output or f"final-{args.provider}.jsonl"

    # Load and Fix Dataset Paths
    dataset_path = project_root / "dataset" / "300_test_set.pkl"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "rb") as f:
        test_dataset = pkl.load(f)
    
    for item in test_dataset:
        original_path = Path(item["image_path"])
        item["image_path"] = str(project_root / "dataset" / "test" / original_path.name)

    # Initialize Components
    print(f"Initializing Multi-Agent System using {args.provider.upper()}...")
    llm_manager = LLMManager(provider=args.provider)
    app = create_diagnosis_graph(llm_manager)

    # Execute Batch Evaluation
    evaluator = BatchEvaluator(
        app=app,
        out_dir=str(results_dir),
        out_file=out_file,
        resume=args.resume
    )
    
    evaluator.run(test_dataset)

if __name__ == "__main__":
    main()
