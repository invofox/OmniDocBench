import logging
import os
import sys

from kpis.comparison_tool import calculate_overall_kpis, generate_kpis

implementation_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "delete"
)
sys.path.append(implementation_path)
from delete.preprocess.main import run as run_preprocess

logger = logging.getLogger(__name__)


def generate_kpi_metrics(predictions, ground_truth):
    preprocess_result = run_preprocess(ground_truth)
    returned = generate_kpis(predictions, preprocess_result, logger)
    return returned
