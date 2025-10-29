import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic.json import pydantic_encoder


def setup_import_paths() -> None:
    """Set up import paths for project modules."""
    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent  # scripts/
    OmniDocBench_dir = scripts_dir.parent  # OmniDocBench/

    # Add the OmniDocBench directory to sys.path for src imports
    OmniDocBench_str = str(OmniDocBench_dir)
    if OmniDocBench_str not in sys.path:
        sys.path.insert(0, OmniDocBench_str)

    # Verify the delete directory exists
    src_dir = OmniDocBench_dir / "delete"
    if not src_dir.exists():
        raise ImportError(f"Expected src directory not found at: {src_dir}")


# Set up paths and import modules
try:
    from delete.commons.utils import configure_logger
    from kpis.comparison_tool import generate_kpis
    from scripts.utils import discover_documents  # type: ignore

    # from kpis.utils import export_experiment_metrics_to_json, new_experiment_id
except ImportError:
    # If imports fail, set up paths and try again
    setup_import_paths()
    from delete.commons.utils import configure_logger
    from kpis.comparison_tool import generate_kpis
    from scripts.utils import discover_documents  # type: ignore

    # from kpis.utils import export_experiment_metrics_to_json, new_experiment_id

# Create logger
LOGGER = configure_logger(__name__, True)
LOGGER.info(f"Initializing {Path(__file__).stem}")


def save_results(results: Any, filepath: str) -> None:
    """Save results to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, default=pydantic_encoder, indent=4)
    except Exception as e:
        LOGGER.error(f"Failed to save results to {filepath}: {e}", exc_info=True)
        raise


def run(
    origin_data_url: str,
    postprocess_result_path: str,
    errors_result_path: str,
    total_operational_metrics_path: str,
) -> None:
    """Generate KPIs from dataset and postprocess result and save metrics files."""
    try:
        # Discover documents in the dataset
        document_paths = discover_documents(origin_data_url, subdirectory="gt")
        total_documents = len(document_paths)

        if total_documents == 0:
            LOGGER.warning(f"No documents found in {origin_data_url}")
            return

        # Load Ground Truth from each document of origin_data_url (assume ground_truth.json file in each document folder)
        ground_truth = {}
        for document_path in document_paths:
            if os.path.exists(os.path.join(document_path, "ground_truth.json")):
                with open(os.path.join(document_path, "ground_truth.json"), "r") as f:
                    ground_truth[Path(document_path).name] = json.load(f)
            else:
                LOGGER.warning(f"No ground truth file found for {document_path}")

        # Load postprocess result
        LOGGER.info("Loading postprocess result...")
        with open(postprocess_result_path, "r") as f:
            final_result_data = json.load(f)

        # Load errors result
        LOGGER.info("Loading errors result...")
        with open(errors_result_path, "r") as f:
            errors_data = json.load(f)

        # Load total operational metrics
        LOGGER.info("Loading total operational metrics...")
        with open(total_operational_metrics_path, "r") as f:
            total_operational_metrics = json.load(f)

        # Create postprocess_obj with original_features and final_result
        postprocess_obj = {
            "ground_truth": ground_truth,
            "execution_result": {},  # Empty execution result
            "final_result": final_result_data,
            "errors_data": errors_data,
            "total_operational_metrics": total_operational_metrics,
        }

        # Generate KPIs
        LOGGER.info("Generating kpis...")
        kpis_result = generate_kpis(postprocess_obj, LOGGER)

        # Save kpis_result.json to the parent of postprocess_result_path
        outputs_path = os.path.dirname(postprocess_result_path)
        kpis_result_path = os.path.join(outputs_path, "metrics.json")
        save_results(kpis_result, kpis_result_path)

        LOGGER.info(f"KPI generation completed. Results saved in: {kpis_result_path}")
        return

    except Exception as e:
        LOGGER.error(f"Error during KPI generation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KPI Generation Only script")
    parser.add_argument(
        "--origin_data_url",
        type=str,
        required=True,
        help="Path to dataset folder",
    )
    parser.add_argument(
        "--postprocess_result_path",
        type=str,
        required=False,
        default="outputs/postprocess_result.json",
        help="Path to postprocess_result.json file",
    )
    parser.add_argument(
        "--errors_result_path",
        type=str,
        required=False,
        default="outputs/errors.json",
        help="Path to errors.json file",
    )
    parser.add_argument(
        "--total_operational_metrics",
        type=str,
        required=False,
        default="outputs/total_operational_metrics.json",
        help="Path to total_operational_metrics.json file",
    )
    args = parser.parse_args()

    LOGGER.info(
        (
            "Starting KPI generation with arguments:\n"
            "Origin data URL: %s\n"
            "Postprocess result path: %s\n"
            "Errors result path: %s\n"
            "Total operational metrics path: %s"
        ),
        args.origin_data_url,
        args.postprocess_result_path,
        args.errors_result_path,
        args.total_operational_metrics,
    )

    run(
        args.origin_data_url,
        args.postprocess_result_path,
        args.errors_result_path,
        args.total_operational_metrics,
    )
