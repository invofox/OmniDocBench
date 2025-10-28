import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from joblib import dump, load  # type: ignore
from pydantic.json import pydantic_encoder


class CacheManager:
    """
    Manages multi-level caching for the document processing pipeline.

    Provides caching at three levels:
    1. Document-level: Cache results for each document
    2. Phase-level: Cache preprocess, inference, and postprocess results
    3. Batch-level: Cache batch processing results

    Uses joblib for efficient caching with content-based hashing.
    """

    def __init__(self, cache_dir: str = "cache", enable_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache

        if self.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)
            LOGGER.info(f"Initialized joblib cache in: {self.cache_dir}")
        else:
            LOGGER.info("Caching disabled")

    def _get_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key from arguments"""
        # Create a stable string representation
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Get cache file path for a given type and key"""
        return self.cache_dir / cache_type / f"{key}.pkl"

    def get_cached_result(self, cache_type: str, key: str) -> Optional[Any]:
        """Retrieve cached result if available"""
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(cache_type, key)
        if cache_path.exists():
            try:
                return load(cache_path)
            except Exception as e:
                LOGGER.warning(f"Failed to load cache from {cache_path}: {e}")
                return None
        return None

    def save_cached_result(self, cache_type: str, key: str, result: Any) -> bool:
        """Save result to cache"""
        if not self.enable_cache:
            return False

        cache_path = self._get_cache_path(cache_type, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dump(result, cache_path)
            LOGGER.debug(f"Cached result to: {cache_path}")
            return True
        except Exception as e:
            LOGGER.warning(f"Failed to save cache to {cache_path}: {e}")
            return False

    def get_document_cache_key(
        self,
        document_path: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate cache key for a document"""
        # Include file modification times for cache invalidation
        doc_mtime = os.path.getmtime(document_path) if os.path.exists(document_path) else 0

        key_data = {
            "document_path": document_path,
            "config": config,
            "doc_mtime": doc_mtime,
        }
        return self._get_cache_key(**key_data)

    def get_phase_cache_key(self, phase: str, document_id: str, inputs: Any) -> str:
        """Generate cache key for a processing phase"""
        key_data = {
            "phase": phase,
            "document_id": document_id,
            "inputs": str(inputs),  # Convert to string for hashing
        }
        return self._get_cache_key(**key_data)


# Global cache manager instance
cache_manager: Optional[CacheManager] = None


def setup_cache_manager(cache_dir: str = "cache", enable_cache: bool = True) -> CacheManager:
    """Initialize the global cache manager"""
    global cache_manager
    cache_manager = CacheManager(cache_dir, enable_cache)
    return cache_manager


def setup_import_paths() -> None:
    """Set up import paths for project modules."""
    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent  # scripts/
    OmniDocBench_dir = scripts_dir.parent  # OmniDocBench/

    # Add the OmniDocBench directory to sys.path for OmniDocBench imports
    OmniDocBench_str = str(OmniDocBench_dir)
    if OmniDocBench_str not in sys.path:
        sys.path.insert(0, OmniDocBench_str)

    # Verify the delete directory exists
    src_dir = OmniDocBench_dir / "delete"
    if not src_dir.exists():
        raise ImportError(f"Expected src directory not found at: {src_dir}")


# Set up paths and import modules
try:
    from delete.commons.utils import Timer, configure_logger
    from delete.inference.main import run as run_inference
    from delete.postprocess.main import run as run_postprocess
    from delete.preprocess.main import run as run_preprocess
    from scripts.utils import discover_documents  # type: ignore
except ImportError:
    # If imports fail, set up paths and try again
    setup_import_paths()
    from delete.commons.utils import Timer, configure_logger
    from delete.inference.main import run as run_inference
    from delete.postprocess.main import run as run_postprocess
    from delete.preprocess.main import run as run_preprocess
    from scripts.utils import discover_documents  # type: ignore

# Create logger
LOGGER = configure_logger(__name__, True)
LOGGER.info("Initializing runtime")


def save_results(results: Any, filepath: str, artifact_paths: list[str] | None = None) -> None:
    """Save results to a JSON file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, default=pydantic_encoder, indent=4)
        LOGGER.debug(f"Results saved successfully to {filepath}")
        # Track artifact path if list is provided
        if artifact_paths is not None:
            artifact_paths.append(filepath)
    except Exception as e:
        LOGGER.error(f"Failed to save results to {filepath}: {e}", exc_info=True)
        raise


def save_object(obj: Any, filepath: str, artifact_paths: list[str] | None = None) -> None:
    """Save an object to a file using joblib."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        dump(obj, filepath)
        LOGGER.debug(f"Object saved successfully to {filepath}")
        # Track artifact path if list is provided
        if artifact_paths is not None:
            artifact_paths.append(filepath)
    except Exception as e:
        LOGGER.error(f"Failed to save object to {filepath}: {e}", exc_info=True)
        raise


def safe_key(key: str) -> str:
    """
    Sanitize a key to make it safe for use in file paths.
    Replaces unsafe characters (forward and backward slashes) with underscores
    to prevent issues with file system paths.
    """
    return key.replace("/", "_").replace("\\", "_")


def build_preprocess_kwargs_for_document(
    document_path: str,
    region: str | None,
) -> dict[str, Any]:
    """Build kwargs for the new preprocess.run function for a single document."""
    kwargs: dict[str, Any] = {}
    kwargs["input_url"] = document_path
    kwargs["region"] = region

    LOGGER.debug(f"kwargs built for {document_path}: {kwargs}")

    return kwargs

def process_batch(
    batch_docs: list[str],
    configuration: dict[str, Any],
    region: str | None,
) -> list[dict[str, Any]]:
    """Process a batch of documents in parallel with individual document caching."""
    results = []
    cached_results = []
    documents_to_process = []

    # Check cache for each document first
    if cache_manager and cache_manager.enable_cache:
        for doc_path in batch_docs:
            document_id = os.path.basename(doc_path)
            doc_cache_key = cache_manager.get_document_cache_key(doc_path, configuration)
            cached_result = cache_manager.get_cached_result("document", doc_cache_key)

            if cached_result is not None:
                LOGGER.info(f"Using cached result for document: {document_id}")
                cached_results.append(cached_result)
            else:
                documents_to_process.append(doc_path)
    else:
        documents_to_process = batch_docs

    # Add cached results to final results
    results.extend(cached_results)

    if documents_to_process:
        LOGGER.info(
            f"Processing {len(documents_to_process)} uncached documents in batch "
            f"(skipped {len(cached_results)} cached documents)"
        )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    process_single_document,
                    doc_path,
                    os.path.basename(doc_path),
                    configuration,
                    region,
                ): doc_path
                for doc_path in documents_to_process
            }

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    doc_path = futures[future]
                    document_id = os.path.basename(doc_path)
                    LOGGER.error(f"Error processing document {document_id}: {e}")
                    results.append(
                        {
                            "document_id": document_id,
                            "document_path": doc_path,
                            "error": str(e),
                            "final_result": {},
                            "errors": {
                                document_id: {
                                    "error": str(e),
                                    "document_path": doc_path,
                                }
                            },
                        }
                    )
    else:
        LOGGER.info(f"All {len(cached_results)} documents in batch were cached - no processing needed")

    return results


def cached_preprocess_phase(document_id: str, kwargs: Dict[str, Any]) -> Optional[Any]:
    """Cached preprocessing phase"""
    if not cache_manager or not cache_manager.enable_cache:
        return None

    cache_key = cache_manager.get_phase_cache_key("preprocess", document_id, kwargs)
    cached_result = cache_manager.get_cached_result("preprocess", cache_key)

    if cached_result is not None:
        LOGGER.info(f"Using cached preprocessing result for document: {document_id}")
        return cached_result

    try:
        result = run_preprocess(**kwargs)
        cache_manager.save_cached_result("preprocess", cache_key, result)
        return result
    except Exception as e:
        LOGGER.error(f"Preprocessing failed for document {document_id}: {e}")
        raise


def cached_inference_phase(document_id: str, feature: Any) -> Optional[Any]:
    """Cached inference phase"""
    if not cache_manager or not cache_manager.enable_cache:
        return None

    # Create a simplified cache key based on feature content
    cache_key = cache_manager.get_phase_cache_key(
        "inference",
        document_id,
        {"feature_id": getattr(feature, "id", str(feature))},
    )
    cached_result = cache_manager.get_cached_result("inference", cache_key)

    if cached_result is not None:
        LOGGER.info(f"Using cached inference result for document: {document_id}")
        return cached_result

    try:
        result = run_inference(feature)
        cache_manager.save_cached_result("inference", cache_key, result)
        return result
    except Exception as e:
        LOGGER.error(f"Inference failed for document {document_id}: {e}")
        raise


def cached_postprocess_phase(
    document_id: str, inference_result: Any, inference_metrics: Any, feature: Any
) -> Optional[Any]:
    """Cached postprocessing phase"""
    if not cache_manager or not cache_manager.enable_cache:
        return None

    # Create a simplified cache key based on inputs
    cache_key = cache_manager.get_phase_cache_key(
        "postprocess",
        document_id,
        {"inference_result_id": getattr(inference_result, "id", str(inference_result))},
    )
    cached_result = cache_manager.get_cached_result("postprocess", cache_key)

    if cached_result is not None:
        LOGGER.info(f"Using cached postprocessing result for document: {document_id}")
        return cached_result

    try:
        result = run_postprocess(inference_result, inference_metrics, feature)
        cache_manager.save_cached_result("postprocess", cache_key, result)
        return result
    except Exception as e:
        LOGGER.error(f"Postprocessing failed for document {document_id}: {e}")
        raise


def process_single_document(
    document_path: str,
    document_id: str,
    config: dict[str, Any],
    region: str | None,
) -> dict[str, Any]:
    """Process a single document through the complete pipeline with phase-level caching."""
    LOGGER.info(f"Processing document: {document_id}")

    try:
        # Check if we have a complete cached result for this document
        if cache_manager and cache_manager.enable_cache:
            doc_cache_key = cache_manager.get_document_cache_key(document_path, config)
            cached_doc_result = cache_manager.get_cached_result("document", doc_cache_key)

            if cached_doc_result is not None:
                LOGGER.info(f"Using cached complete result for document: {document_id}")
                return cached_doc_result if isinstance(cached_doc_result, dict) else {}

        # Preprocess single document with caching
        kwargs = build_preprocess_kwargs_for_document(document_path, region)

        result = (
            cached_preprocess_phase(document_id, kwargs)
            if cache_manager and cache_manager.enable_cache
            else run_preprocess(**kwargs)
        )

        if result is None:
            raise ValueError("Preprocessing failed, cannot proceed.")

        feature = result
        LOGGER.info(f"Preprocessing completed for document: {document_id}")

        # Inference single document with caching
        result = (
            cached_inference_phase(document_id, feature)
            if cache_manager and cache_manager.enable_cache
            else run_inference(feature)
        )

        if result is None:
            raise ValueError("Inference failed, cannot proceed.")

        inference_result, inference_metrics = result
        LOGGER.info(f"Inference completed for document: {document_id}")

        # Postprocess single document with caching
        final_result = (
            cached_postprocess_phase(document_id, inference_result, inference_metrics, feature)
            if cache_manager and cache_manager.enable_cache
            else run_postprocess(inference_result, inference_metrics, feature)
        )

        if final_result is None:
            raise ValueError("Postprocessing failed, cannot proceed.")

        LOGGER.info(f"Postprocessing completed for document: {document_id}")

        document_result = {
            "document_id": document_id,
            "document_path": document_path,
            "inference_result": inference_result,
            "final_result": {"extraction": final_result["extraction"]},
            "metrics": final_result["metrics"],
        }

        # Cache the complete document result
        if cache_manager and cache_manager.enable_cache:
            doc_cache_key = cache_manager.get_document_cache_key(document_path, config)
            cache_manager.save_cached_result("document", doc_cache_key, document_result)

        return document_result

    except Exception as e:
        LOGGER.error(f"Error processing document {document_id}: {e}")
        return {
            "document_id": document_id,
            "document_path": document_path,
            "error": str(e),
            "final_result": {},
            "errors": {document_id: {"error": str(e), "document_path": document_path}},
        }


def run(
    origin_data_url: str,
    output_dir: str,
    region: str | None = None,
    cache_dir: str = "cache",
    enable_cache: bool = True,
) -> list[str]:
    """Run the entire pipeline: process each document individually through preprocess, inference and postprocess using
    new structure."""
    artifact_paths: list[str] = []  # List to store paths of generated artifacts
    try:
        # Initialize cache manager
        setup_cache_manager(cache_dir, enable_cache)

        os.makedirs(output_dir, exist_ok=True)
        LOGGER.info("Starting the OmniDocBench pipeline with new structure.")

        if enable_cache:
            LOGGER.info(f"Using joblib-based caching in: {cache_dir}")
        else:
            LOGGER.info("Caching disabled")

        timing_info = {}

        # Load configuration or use default
        configuration: dict = {}

        # Discover documents in the dataset
        document_paths = discover_documents(origin_data_url, subdirectory="input")
        total_documents = len(document_paths)

        if total_documents == 0:
            LOGGER.warning(f"No documents found in {origin_data_url}")
            return []

        LOGGER.info(f"Found {total_documents} documents to process in {origin_data_url}")

        # Process documents in batches with enhanced caching
        batch_size = 11
        LOGGER.info(f"Processing documents as parallel batches of size {batch_size}")
        document_results = []

        with Timer() as total_timer:
            for i in range(0, total_documents, batch_size):
                batch = document_paths[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_documents + batch_size - 1) // batch_size
                LOGGER.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")

                batch_results = process_batch(batch, configuration, region)
                document_results.extend(batch_results)

        # Combine results from all documents
        LOGGER.info("Combining results from all documents...")

        timing_info.update(
            {
                "total_processing_time_seconds": total_timer.elapsed_time,
                "documents_processed": len(document_results),
                "average_time_per_document": total_timer.elapsed_time / len(document_results)
                if document_results
                else 0,
            }
        )

        # Save combined results
        postprocess_results = {}
        operational_metrics = {}

        for document_result in document_results:
            doc_id = document_result["document_id"]
            postprocess_results[doc_id] = document_result["final_result"]
            operational_metrics[doc_id] = document_result["metrics"]

        results_to_save = {
            "postprocess_result": postprocess_results,
            "operational_metrics": operational_metrics,
        }

        for name, result in results_to_save.items():
            safe_filename = f"{name}.json"
            save_results(
                result,
                os.path.join(output_dir, safe_filename),
                artifact_paths,
            )
            LOGGER.info(f"Saved {safe_filename} to output {output_dir}.")

        LOGGER.info(f"All documents processed in {total_timer.elapsed_time:.2f} seconds")

        # Update timing information for the new document-by-document processing
        timing_info["timestamp"] = time.time()
        total_time = timing_info.get("total_processing_time_seconds", 0)

        # Save timing information
        save_results(
            timing_info,
            os.path.join(output_dir, "timing_results.json"),
            artifact_paths,
        )
        LOGGER.info(f"Total pipeline time: {total_time:.2f} seconds")
        LOGGER.info(f"Processed {timing_info.get('documents_processed', 0)} documents")
        LOGGER.info(f"Average time per document: {timing_info.get('average_time_per_document', 0):.2f} seconds")

        LOGGER.debug("List of artifacts: %s", artifact_paths)
        return artifact_paths

    except Exception as e:
        LOGGER.error(f"Error during local run: {e}", exc_info=True)
        raise e
    finally:
        LOGGER.info("Pipeline completed. Results saved in directory: %s", output_dir)
        LOGGER.info(f"A total of {len(artifact_paths)} artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local run script")
    parser.add_argument(
        "--origin_data_url",
        type=str,
        required=True,
        help="Path to load data from (local path or S3 URL)",
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Region to use: US or EU",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="outputs",
        help="Directory to save output files. Defaults to 'outputs'.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default="cache",
        help="Directory for cache files. Defaults to 'cache'.",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disable all caching functionality.",
    )
    args = parser.parse_args()

    LOGGER.info(
        (
            "Starting local inference run with arguments:\n"
            "Origin data URL: %s\n"
            "Output directory: %s\n"
            "Region: %s\n"
            "Cache directory: %s\n"
            "Cache enabled: %s"
        ),
        args.origin_data_url,
        args.output_dir,
        args.region,
        args.cache_dir,
        not args.disable_cache,
    )
    artifact_paths = run(
        args.origin_data_url,
        args.output_dir,
        args.region,
        args.cache_dir,
        not args.disable_cache,
    )
