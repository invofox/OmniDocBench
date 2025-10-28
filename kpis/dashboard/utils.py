"""
Utility functions for the dashboard to handle data processing and analysis.
"""

import datetime
import json
import os
from typing import Any

import pandas as pd  # type: ignore
from dvc.api import read, params_show  # type: ignore
from git import Repo as GitRepo  # type: ignore
from pandas.io.formats import style  # type: ignore
from config import ( # type: ignore
    COLOR_PALETTE,
    COST_RANKING_KEY_METRICS,
    COST_RANKING_SORT_COLUMNS,
    FILES,
    NESTED_FIELDS,
    PERFORMANCE_RANKING_KEY_METRICS,
    PERFORMANCE_RANKING_SORT_COLUMNS,
    TIME_RANKING_KEY_METRICS,
    TIME_RANKING_SORT_COLUMNS,
)

from delete.commons.utils import configure_logger

# Create logger for this module
LOGGER = configure_logger(__name__, log_to_stdout=True)

# Initialize Git repository for metadata extraction
git_repo = GitRepo(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def load_experiment_summary(experiment_name: str, experiment_commit: str) -> dict[str, Any] | None:
    """
    Load experiment summary data including metrics and configuration.

    Args:
        experiment_name: Name of the DVC experiment
        experiment_commit: Git commit SHA for the experiment

    Returns:
        Dictionary containing experiment data or None if not found
    """
    def iterative_assign(base_dict: dict, keys: list[str], value: Any) -> dict:
        """Helper function to assign values in nested dictionaries."""
        if isinstance(value, dict):
            for k, v in value.items():
                base_dict = iterative_assign(base_dict, keys + [k], v)
        elif isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            if ".".join(keys) not in base_dict:
                base_dict[".".join(keys)] = []
            base_dict[".".join(keys)].extend(value)
        elif isinstance(value, (int, float)):
            if ".".join(keys) not in base_dict:
                base_dict[".".join(keys)] = []
            base_dict[".".join(keys)].append(value)
        return base_dict

    data = {}

    # Define all files to load from outputs directory
    output_files = FILES

    for key, filename in output_files.items():
        file_path = f"outputs/{filename}"
        try:
            # Use DVC API to read the file
            file_content = read(file_path, rev=experiment_name)
            file_data = json.loads(file_content)
            # Special handling for metrics to flatten nested structure
            if key in NESTED_FIELDS:
                mean_data: dict = {}
                for _, metrics in file_data.items():
                    for metric_name, metric_value in metrics.items():
                        mean_data = iterative_assign(mean_data, [metric_name], metric_value)
                file_data = {f"{k}.mean": sum(v) / len(v) if len(v) > 0 else 0 for k, v in mean_data.items()}
                file_data.update({f"{k}.total": sum(v) for k, v in mean_data.items()})



            # For other files, store under their respective keys
            data[key] = file_data

        except Exception as e:
            LOGGER.error(f"Error reading {filename} for {experiment_name}: {e}")
            continue

    data["commit"] = experiment_commit
    data["timestamp"] = git_repo.commit(experiment_commit).committed_date
    data["dvc_params"] = params_show(rev=experiment_name)

    return data if data else None


def extract_configuration_matrix(experiment_name: str, experiment_data: dict[str, Any]) -> pd.DataFrame:
    """
    Extract the configuration matrix from experiment data.

    Args:
        experiment_data: Experiment data dictionary

    Returns:
        DataFrame with configuration details for each experiment run
    """
    config_rows = []

    # Load timestamp information
    timestamp = datetime.datetime.fromtimestamp(
        experiment_data.get("timestamp", 0)
    ).strftime("%Y-%m-%d %H:%M:%S")
    row = {"Timestamp": timestamp}
    row["Name"] = experiment_name

    # Extract git metadata from commit
    commit_sha = experiment_data.get("commit", "")
    user, git_branch = extract_git_metadata(commit_sha)

    # Add common fields
    row["Commit (Short SHA)"] = commit_sha[:7]
    row["User"] = user
    row["Git Branch"] = git_branch
    row.update(experiment_data.get("dvc_params", {}))

    # Add here any other relevant configuration details

    config_rows.append(row)

    return pd.DataFrame(config_rows)


def format_metric_value(value: Any, metric_name: str) -> str:
    """
    Format metric values for display.

    Args:
        value: The metric value
        metric_name: Name of the metric for context

    Returns:
        Formatted string representation
    """
    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):
        # Document/count formatting (integer values)
        if "documents" in metric_name.lower() or "calls" in metric_name.lower() or "pages" in metric_name.lower():
            return f"{int(value):,}"
        # Cost formatting
        if "cost" in metric_name.lower():
            return f"${value:.4f}"
        # Time/latency formatting
        if "latency" in metric_name.lower() or "time" in metric_name.lower():
            return f"{value:.2f}s"
        # Percentage formatting
        if "rate" in metric_name.lower() or "ratio" in metric_name.lower():
            return f"{value:.2%}"
        # Default numerical formatting
        # Debug: Log cases that fall through to default formatting
        LOGGER.debug(f"Using default formatting for metric '{metric_name}' with value {value}")
        return f"{value:.3f}"

    return str(value)


def create_performance_ranking(experiments: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Create a performance ranking across experiments based on accuracy metrics.
    Results are sorted by performance (higher is better): end_to_end.ratio_accurate_docs,
    then end_to_end.ratio_accurate_fields, then extraction.ratio_accurate_docs,
    then extraction.ratio_accurate_fields.

    Args:
        experiments: Dictionary of experiment data

    Returns:
        DataFrame with performance metrics sorted by performance (best first)
    """
    ranking_data = []

    for exp_name, exp_data in experiments.items():
        # Get summary metrics
        processed_metrics = exp_data.get("summary", {}).get("processed", {})
        end_to_end_metrics = exp_data.get("summary", {}).get("end_to_end", {})
        error_info_metrics = exp_data.get("error_info", {})
        custom_metrics = exp_data.get("metrics", {})

        row: dict[str, Any] = {
            "experiment": os.path.basename(exp_name),
            "experiment_path": exp_name,
        }

        # Key metrics for performance ranking
        performance_key_metrics = PERFORMANCE_RANKING_KEY_METRICS

        for metric in performance_key_metrics:
            if metric in end_to_end_metrics:
                row["end_to_end." + metric] = end_to_end_metrics[metric]
            elif metric in processed_metrics:
                row["processed." + metric] = processed_metrics[metric]
            elif metric in error_info_metrics:
                row[metric] = error_info_metrics[metric]
            elif metric in custom_metrics:
                row[metric] = custom_metrics[metric]
            else:
                row[metric] = 0.0

        ranking_data.append(row)

    if not ranking_data:
        return pd.DataFrame()

    df = pd.DataFrame(ranking_data)

    # Sort by performance metrics (higher is better)
    sort_columns = PERFORMANCE_RANKING_SORT_COLUMNS

    # Only sort by columns that exist in the dataframe
    existing_sort_columns = [col for col in sort_columns if col in df.columns]

    if existing_sort_columns:
        df = df.sort_values(existing_sort_columns, ascending=False)

    # Format the values for better display
    for col in df.columns:
        if col not in ["experiment", "experiment_path"] and col in df.columns:
            df[col + "_formatted"] = df[col].apply(lambda x, col=col: format_metric_value(x, col))

    return df


def create_cost_ranking(experiments: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Create a cost ranking across experiments.
    Results are sorted by cost (lower is better): operations.average_cost_per_document,
    then operations.total_cost.

    Args:
        experiments: Dictionary of experiment data

    Returns:
        DataFrame with cost metrics sorted by cost efficiency (best first)
    """
    ranking_data = []

    for exp_name, exp_data in experiments.items():
        row: dict[str, Any] = {
            "experiment": os.path.basename(exp_name),
            "experiment_path": exp_name,
        }

        # Key metrics for cost ranking (in order of relevance)
        key_metrics = COST_RANKING_KEY_METRICS

        for metric in key_metrics:
            if metric in exp_data.get("summary", {}):
                row[metric] = exp_data.get("summary", {}).get(metric)
            elif metric in exp_data.get("operations", {}):
                row[metric] = exp_data.get("operations", {}).get(metric)
            elif metric in exp_data.get("metrics", {}):
                row[metric] = exp_data.get("metrics", {}).get(metric)
            else:
                row[metric] = 0.0

        ranking_data.append(row)

    if not ranking_data:
        return pd.DataFrame()

    df = pd.DataFrame(ranking_data)

    # Sort by cost metrics (lower is better)
    sort_columns = COST_RANKING_SORT_COLUMNS

    # Only sort by columns that exist in the dataframe
    existing_sort_columns = [col for col in sort_columns if col in df.columns]

    if existing_sort_columns:
        df = df.sort_values(existing_sort_columns, ascending=True)

    # Format the values for better display
    for col in df.columns:
        if col not in ["experiment", "experiment_path"] and col in df.columns:
            df[col + "_formatted"] = df[col].apply(lambda x, col=col: format_metric_value(x, col))

    return df


def create_time_ranking(experiments: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Create a time/latency ranking across experiments.
    Results are sorted by time efficiency (lower is better): operations.average_latency_per_document,
    then operations.total_latency_seconds.

    Args:
        experiments: Dictionary of experiment data

    Returns:
        DataFrame with time metrics sorted by time efficiency (best first)
    """
    ranking_data = []

    for exp_name, exp_data in experiments.items():
        # Get summary metrics
        row: dict[str, Any] = {
            "experiment": os.path.basename(exp_name),
            "experiment_path": exp_name,
        }

        # Key metrics for time ranking (in order of relevance)
        key_metrics = TIME_RANKING_KEY_METRICS

        for metric in key_metrics:
            if metric in exp_data.get("summary", {}):
                row[metric] = exp_data.get("summary", {}).get(metric)
            elif metric in exp_data.get("operations", {}):
                row[metric] = exp_data.get("operations", {}).get(metric)
            elif metric in exp_data.get("metrics", {}):
                row[metric] = exp_data.get("metrics", {}).get(metric)
            else:
                row[metric] = 0.0

        ranking_data.append(row)

    if not ranking_data:
        return pd.DataFrame()

    df = pd.DataFrame(ranking_data)

    # Sort by time metrics (lower is better)
    sort_columns = TIME_RANKING_SORT_COLUMNS

    # Only sort by columns that exist in the dataframe
    existing_sort_columns = [col for col in sort_columns if col in df.columns]

    if existing_sort_columns:
        df = df.sort_values(existing_sort_columns, ascending=True)

    # Format the values for better display
    for col in df.columns:
        if col not in ["experiment", "experiment_path"] and col in df.columns:
            df[col + "_formatted"] = df[col].apply(lambda x, col=col: format_metric_value(x, col))

    return df


def extract_combined_configuration_matrix(
    experiments: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Extract the configuration matrix from multiple experiments.

    Args:
        experiments: Dictionary of experiment data with experiment paths as keys

    Returns:
        DataFrame with configuration details for each experiment
    """
    combined_config_rows = []

    for exp_path, exp_data in experiments.items():
        exp_name = os.path.basename(exp_path)

        # Load timestamp information
        timestamp = datetime.datetime.fromtimestamp(exp_data.get("timestamp", 0)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Extract git metadata from commit
        commit_sha = exp_data.get("commit", "")
        user, git_branch = extract_git_metadata(commit_sha)

        # Add common fields
        row = {
            "Timestamp": timestamp,
            "Commit (Short SHA)": commit_sha[:7],
            "User": user,
            "Git Branch": git_branch,
            "Experiment": exp_name,
        }

        row.update(exp_data.get("dvc_params", {}))

        # Add here any other relevant configuration details
        combined_config_rows.append(row)

    return pd.DataFrame(combined_config_rows)


def generate_experiment_colors(experiments: dict[str, Any]) -> dict[str, str]:
    """
    Generate consistent colors for experiments to be used across all tables.

    Args:
        experiments: Dictionary of experiment data with experiment paths as keys

    Returns:
        Dictionary mapping experiment names to color codes
    """

    experiment_names = [os.path.basename(exp_path) for exp_path in experiments]
    colors = {}

    for i, exp_name in enumerate(experiment_names):
        colors[exp_name] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

    return colors


def style_dataframe_with_colors(
    df: pd.DataFrame, colors: dict[str, str], experiment_column: str = "experiment"
) -> pd.DataFrame | style.Styler:
    """
    Apply row coloring to a dataframe based on experiment names.

    Args:
        df: DataFrame to style
        colors: Dictionary mapping experiment names to color codes
        experiment_column: Name of the column containing experiment names

    Returns:
        Styled DataFrame
    """
    if df.empty or experiment_column not in df.columns:
        return df

    def highlight_rows(row: pd.Series) -> list[str]:
        exp_name = row[experiment_column]
        color = colors.get(exp_name, "#FFFFFF")  # Default to white if not found
        return [f"background-color: {color}"] * len(row)

    return df.style.apply(highlight_rows, axis=1)


def style_config_matrix_with_colors(df: pd.DataFrame, colors: dict[str, str]) -> pd.DataFrame | style.Styler:
    """
    Apply row coloring to configuration matrix based on experiment names.

    Args:
        df: Configuration matrix DataFrame to style
        colors: Dictionary mapping experiment names to color codes

    Returns:
        Styled DataFrame
    """
    if df.empty or "Experiment" not in df.columns:
        return df

    def highlight_rows(row: pd.Series) -> list[str]:
        exp_name = row["Experiment"]
        color = colors.get(exp_name, "#FFFFFF")  # Default to white if not found
        return [f"background-color: {color}"] * len(row)

    return df.style.apply(highlight_rows, axis=1)


# Operational metric name mapping for better display
OPERATIONAL_METRIC_NAMES = {
    "total_documents": "Total Documents",
    "documents_with_latency_metrics": "Documents with Latency Metrics",
    "documents_with_cost_metrics": "Documents with Cost Metrics",
    "total_latency_seconds": "Total Latency",
    "average_latency_per_document": "Avg Latency per Doc",
    "total_prompt_tokens_cost": "Total Prompt Tokens Cost",
    "average_prompt_tokens_cost_per_document": "Avg Prompt Tokens Cost per Doc",
    "total_completion_tokens_cost": "Total Completion Tokens Cost",
    "average_completion_tokens_cost_per_document": "Avg Completion Tokens Cost per Doc",
    "total_total_cost": "Total Tokens Cost",
    "average_total_cost_per_document": "Avg Total Tokens per Doc",
    "total_cost": "Total Cost",
    "average_cost_per_document": "Avg Cost per Doc",
    "average_cost_per_page": "Avg Cost per Page",
}


def format_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format column headers to be more human-readable.

    Args:
        df: DataFrame to format

    Returns:
        DataFrame with improved column names
    """
    if df.empty:
        return df

    column_mapping = {
        "end_to_end.ratio_accurate_docs": "End2End Document Accuracy",
        "end_to_end.ratio_accurate_fields": "End2End Field Accuracy",
        "processed.ratio_accurate_docs": "Processed Document Accuracy",
        "processed.ratio_accurate_fields": "Processed Field Accuracy",
        "operations.average_cost_per_document": "Avg Cost per Doc",
        "operations.total_cost": "Total Cost",
        "operations.average_latency_per_document": "Avg Latency per Doc",
        "operations.average_latency_per_page": "Avg Latency per Page",
        "operations.total_latency_seconds": "Total Latency",
        "operations.total_documents": "Total Documents",
        "operations.average_cost_per_page": "Avg Cost per Page",
        "average_latency_per_page": "Avg Latency per Page",
        "success_rate": "Success Rate",
        "error_rate": "Error Rate",
        "experiment": "Experiment",
        "model": "Model",
        "input_images": "Input Images",
        "input_ocr": "Input OCR",
        "work_by_page": "Work by Page",
        "section_focus": "Section Focus",
        "few_shot": "Few-shot",
        "justification": "Justification",
        "region": "Region",
        "field_name": "Field",
        "accuracy": "Accuracy",
        "std_deviation": "Std Dev",
    }

    # Create a copy to avoid modifying the original
    formatted_df = df.copy()

    # Rename columns that exist in our mapping
    existing_columns = [col for col in formatted_df.columns if col in column_mapping]
    rename_dict = {col: column_mapping[col] for col in existing_columns}

    if rename_dict:
        formatted_df = formatted_df.rename(columns=rename_dict)

    # Special handling for operational metrics display names
    for col in formatted_df.columns:
        if col in OPERATIONAL_METRIC_NAMES:
            formatted_df = formatted_df.rename(columns={col: OPERATIONAL_METRIC_NAMES[col]})

    return formatted_df


def enhance_table_styling(styled_df: style.Styler) -> style.Styler:
    """
    Apply enhanced styling to a pandas Styler object for better aesthetics.

    Args:
        styled_df: pandas Styler object

    Returns:
        Enhanced Styler object with improved formatting
    """
    if styled_df is None:
        return styled_df

    # Add CSS styling for better appearance
    return styled_df.set_table_styles(
        [
            # Header styling
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#f8f9fa"),
                    ("color", "#212529"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #dee2e6"),
                    ("padding", "12px 8px"),
                    ("font-size", "14px"),
                ],
            },
            # Cell styling
            {
                "selector": "tbody td",
                "props": [
                    ("text-align", "center"),
                    ("border", "1px solid #dee2e6"),
                    ("padding", "10px 8px"),
                    ("font-size", "13px"),
                ],
            },
            # Table styling
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("margin", "0 auto"),
                    ("width", "100%"),
                    ("box-shadow", "0 2px 4px rgba(0,0,0,0.1)"),
                    ("border-radius", "8px"),
                    ("overflow", "hidden"),
                ],
            },
            # Hover effect
            {
                "selector": "tbody tr:hover",
                "props": [
                    ("background-color", "#f5f5f5 !important"),
                    ("cursor", "pointer"),
                ],
            },
        ]
    )


def apply_enhanced_table_formatting(
    df: pd.DataFrame,
    colors: dict[str, str],
    experiment_column: str = "experiment",
    format_headers: bool = True,
) -> pd.DataFrame | style.Styler:
    """
    Apply comprehensive formatting to a DataFrame including coloring, headers, and styling.

    Args:
        df: DataFrame to format
        colors: Dictionary mapping experiment names to color codes
        experiment_column: Name of the column containing experiment names
        format_headers: Whether to format column headers

    Returns:
        Styled DataFrame with enhanced formatting
    """
    if df.empty:
        return df

    # Format column headers if requested
    if format_headers:
        df = format_column_headers(df)
        # Update experiment_column name if it was changed
        if experiment_column == "experiment" and "Experiment" in df.columns:
            experiment_column = "Experiment"

    # Apply row coloring
    styled_df = style_dataframe_with_colors(df, colors, experiment_column)

    # Apply enhanced styling only if we have a Styler object
    if isinstance(styled_df, style.Styler):
        styled_df = enhance_table_styling(styled_df)

    return styled_df


def get_main_sections_from_fields(field_metrics_df: pd.DataFrame) -> list[str]:
    """
    Extract main sections from field names that have section_avg.

    Args:
        field_metrics_df: DataFrame with field metrics

    Returns:
        List of main section names
    """
    if field_metrics_df.empty:
        return []

    section_avg_fields = field_metrics_df[
        field_metrics_df["field_name"].str.contains("section_avg", case=False, na=False)
    ]

    main_sections = []
    for field_name in section_avg_fields["field_name"]:
        # Extract section name before "_section_avg"
        section_name = field_name.split("_section_avg")[0]
        # Only include if it doesn't contain a dot (which indicates it's a subsection)
        if "." not in section_name:
            main_sections.append(section_name)

    return sorted(set(main_sections))


def extract_llm_requests_data(experiment_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and process LLM requests data from experiment.

    Args:
        experiment_data: Full experiment data dictionary

    Returns:
        Dictionary with processed LLM requests data organized by document
    """
    llm_requests = experiment_data.get("llm_requests", {})

    if not llm_requests:
        return {}

    # Find all documents available
    docs_data = {}
    for key, value in llm_requests.items():
        doc_id = key
        docs_data[doc_id] = value

    return docs_data


def extract_git_metadata(commit_sha: str) -> tuple[str, str]:
    """
    Extract user and git branch information from a commit SHA.

    Args:
        commit_sha: The commit SHA to extract metadata from

    Returns:
        Tuple of (user, git_branch)
    """
    try:
        # Get commit information from Git
        git_commit = git_repo.commit(commit_sha)
        author_name = git_commit.author.name
        author_email = git_commit.author.email
        user = f"{author_name} <{author_email}>"

        # Determine the branches that contain this commit
        branches_output = git_repo.git.branch("--contains", commit_sha, "--all")
        branches = []
        for line in branches_output.split("\n"):
            line = line.strip()
            if line and not line.startswith("*"):
                # Clean up branch names (remove origin/ prefix, etc.)
                branch = line.replace("remotes/origin/", "").replace("origin/", "").strip()
                if branch and branch not in branches and not branch.startswith("("):
                    branches.append(branch)

        # Use the first branch as primary, or "main" as fallback
        git_branch = branches[0] if branches else "main"

        return user, git_branch

    except Exception as e:
        LOGGER.warning(f"Error extracting git metadata for commit {commit_sha}: {e}")
        return "Unknown", "Unknown"
