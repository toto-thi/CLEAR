import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set
from tqdm import tqdm

class BatchEvaluator:
    """
    Orchestrates batch evaluation of the agentic pipeline.
    Handles resume logic, state management, and error reporting.
    """
    def __init__(
        self,
        app: Any,
        out_dir: str,
        out_file: str,
        resume: bool = True,
        log_interval: int = 10
    ):
        self.app = app
        self.out_dir = Path(out_dir)
        self.out_file = self.out_dir / out_file
        self.resume = resume
        self.log_interval = log_interval
        
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_processed_ids(self) -> Set[str]:
        """Scans existing output file to find already processed image IDs."""
        processed_ids = set()
        if self.resume and self.out_file.exists():
            print(f"Scanning for existing results in {self.out_file}...")
            with open(self.out_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        st = json.loads(line)
                        p = st.get("image_path") or ""
                        if p:
                            processed_ids.add(Path(p).stem)
                    except Exception:
                        pass
            print(f"Found {len(processed_ids)} already processed images.")
        return processed_ids

    def _prepare_initial_state(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs the initial LangGraph state for a single case."""
        return {
            "image_path": item["image_path"],
            "patient_data": {
                "age": item.get("age", ""),
                "sex": item.get("sex", ""),
                "lesion_location": item.get("anatom_site", ""),
            },
            "ground_truth": item.get("diagnosis") or item.get("label") or "",
            "trace": [],
        }

    def run(self, test_dataset: List[Dict[str, Any]]) -> Path:
        """Executes the batch evaluation over the provided dataset."""
        print(f"Starting batch evaluation | Dataset: {len(test_dataset)} items")
        
        processed_ids = self._load_processed_ids()
        unprocessed = [
            it for it in test_dataset 
            if Path(it["image_path"]).stem not in processed_ids
        ]
        
        total = len(test_dataset)
        remaining = len(unprocessed)
        print(f"⏭️  To process: {remaining}/{total}")

        if remaining == 0:
            print("All items already processed.")
            return self.out_file

        mode = "a" if self.resume and self.out_file.exists() else "w"
        n_ok, n_fail = 0, 0
        start_time = time.time()

        with open(self.out_file, mode, encoding="utf-8") as jf:
            iterator = tqdm(unprocessed, desc="Evaluating", unit="img")
            for i, item in enumerate(iterator):
                image_stem = Path(item["image_path"]).stem
                try:
                    state = self._prepare_initial_state(item)
                    run_id = int(time.time())
                    config = {"configurable": {"thread_id": f"case-{image_stem}-{run_id}"}}
                    
                    final_state = self.app.invoke(state, config=config)

                    # Ensure metadata is preserved in the output
                    final_state.setdefault("image_path", item["image_path"])
                    final_state.setdefault("ground_truth", state["ground_truth"])

                    jf.write(json.dumps(final_state, ensure_ascii=False) + "\n")
                    jf.flush()
                    n_ok += 1

                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)}"
                    print(f"\n❌ Failed on {image_stem}: {err}")
                    error_state = {
                        **self._prepare_initial_state(item),
                        "_error": err,
                    }
                    jf.write(json.dumps(error_state, ensure_ascii=False) + "\n")
                    jf.flush()
                    n_fail += 1

                # Periodically log ETA
                if (i + 1) % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    avg = elapsed / (i + 1)
                    eta = avg * (remaining - i - 1)
                    iterator.set_postfix({"ETA": f"{int(eta//60)}m {int(eta%60)}s"})

        print("\n" + "="*48)
        print(f"✅ BATCH COMPLETE | Saved to: {self.out_file}")
        print(f"   Success: {n_ok} | Fail: {n_fail}")
        print("="*48)

        return self.out_file
