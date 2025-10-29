from typing import Any
from pandas import DataFrame

COLOR_PALETTE = [
        "#78B9B5",
        "#DCA06D",
        "#F7374F",
        "#065084",
        "#901E3E",
        "#A55B4B",
        "#FFF4B7",
        "#4B4376",
    ]

FILES = {
    "metrics": "metrics.json",
}
NESTED_FIELDS = []

KEY_METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
]

OPERATIONAL_METRICS = [
    "average_cost_per_document",
    "average_latency_per_document",
    "average_latency_per_page",
    "total_cost",
    "total_documents",
    "total_latency_seconds",
    "average_cost_per_page",
]

ACCURACY_METRICS = [
    "metrics.accuracy",
    "metrics.macro_precision",
    "metrics.macro_recall",
]
ERROR_METRICS = []
COST_METRICS = [
    "operations.average_cost_per_document",
    "operations.average_cost_per_page",
    "operations.total_cost",
]
LATENCY_METRICS = [
    "operations.average_latency_per_document",
    "operations.total_latency_seconds",
]

PERFORMANCE_RANKING_KEY_METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
]
# Primary: accuracy, Secondary: macro_precision, Tertiary: macro_recall
PERFORMANCE_RANKING_SORT_COLUMNS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
]

COST_RANKING_KEY_METRICS = [
    "usage.cost.total_cost.total",
    "usage.cost.total_cost.mean",
    "usage.cost.completion_tokens_cost.mean",
    "usage.cost.prompt_tokens_cost.mean",
]
# Primary: total_cost.total, Secondary: total_cost.mean
COST_RANKING_SORT_COLUMNS = ["usage.cost.total_cost.total", "usage.cost.total_cost.mean"]

TIME_RANKING_KEY_METRICS = ["inference_latency.mean", "postprocessing_time.mean"]
# Primary: inference_latency.mean, Secondary: postprocessing_time.mean
TIME_RANKING_SORT_COLUMNS = [
    "inference_latency.mean",
    "postprocessing_time.mean",
]

def format_confusion_matrix(col: Any, title: str, values: list, field_names: list[str]) -> None:
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not isinstance(values[0], list)
        or not isinstance(values[1], list)
    ):
        col.markdown("No confusion matrix available")
        return
    matrix_df = DataFrame(values[0], columns=values[1], index=values[1])
    col.dataframe(
        matrix_df.style.set_caption(title)
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("color", "black"),
                        ("font-size", "16px"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("margin-bottom", "10px"),
                    ],
                }
            ]
        )
        .set_properties(**{"text-align": "center"})
    )

def format_basic(col: Any, title: str, values: list, field_names: list[str]) -> None:
    if len(values) == 1:
        values = values[0]
    col.metric(label=title, value=str(values))


def format_percentage(col: Any, title: str, values: list[float], field_names: list[str]) -> None:
    if len(values) != 1:
        col.markdown(f"Invalid percentage value: {values}")
        return
    percentage_value = f"{values[0]:.2%}"
    col.metric(label=title, value=percentage_value)


DETAILS_CONFIG = {
    "overview": {
        "**📄 Document Accuracy**": {
            "Doc Accuracy (Processed)": {
                "fields": ["accuracy"],
                "format": format_percentage,
            }
        },
        "**⚡ Processing Rates**": {
            "Documents Processed": {
                "fields": ["documents_processed"],
                "format": format_basic,
            }
        },
        "**😵 Confusion Matrix**": {
            "Confusion Matrix": {
                "fields": ["confusion_matrix", "labels"],
                "format": format_confusion_matrix,
            }
        },
    },
    "processing_stats": {
        "Total Documents": {
            "fields": ["documents_processed"],
            "format": format_basic,
        },
    },
}