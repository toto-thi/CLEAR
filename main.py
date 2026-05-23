"""
CLEAR: Critique-Led Explainable Agents with Image Retrieval for Grounded Skin-Lesion Diagnosis.

This is the main entry point for the CLEAR multi-agent diagnostic system.
It provides a unified CLI to run the diagnosis pipeline, initialize the 
Knowledge Base, and perform evaluations.
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CLEAR")

def cmd_run(args):
    """Executes the diagnostic pipeline."""
    logger.info("Starting CLEAR Diagnostic Pipeline...")
    try:
        from experiments.final_pipeline import main as run_pipeline
        sys.argv = [sys.argv[0]] + args.pipeline_args
        run_pipeline()
    except ImportError as e:
        logger.error(f"Failed to import diagnostic pipeline: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
        sys.exit(1)

def cmd_init_kb(args):
    """Initializes the Visual Knowledge Base (VKB)."""
    logger.info("Initializing Visual Knowledge Base...")
    print("\n[NOTE] Knowledge Base initialization is currently managed via Jupyter Notebook.")
    print("Please run 'experiments/create_kb.ipynb' to populate the Qdrant collection.")
    print("Future versions will support CLI-based initialization here.\n")

def cmd_evaluate(args):
    """Runs the evaluation suite."""
    logger.info("Running evaluation suite...")
    print("\n[NOTE] Evaluation is currently performed using notebooks in the 'evaluation/' directory.")
    print("Example: 'evaluation/score_calculator.ipynb'\n")

def main():
    parser = argparse.ArgumentParser(
        description="CLEAR: Critique-Led Explainable Agents for Grounded Skin-Lesion Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run the diagnostic pipeline")
    run_parser.add_argument(
        "pipeline_args", 
        nargs=argparse.REMAINDER, 
        help="Arguments passed to the final-pipeline (e.g., --provider, --output)"
    )
    run_parser.set_defaults(func=cmd_run)

    # 'init-kb' command
    kb_parser = subparsers.add_parser("init-kb", help="Initialize the Knowledge Base")
    kb_parser.set_defaults(func=cmd_init_kb)

    # 'evaluate' command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute the chosen command
    args.func(args)

if __name__ == "__main__":
    main()
