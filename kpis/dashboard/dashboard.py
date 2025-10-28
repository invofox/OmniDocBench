"""
Professional Streamlit Dashboard for Experiment Metrics Analysis and Comparison

This dashboard provides a comprehensive analysis tool for comparing experiment metrics
from the OmniDocBench-delete project. It allows developers to:
- Compare multiple experiments side-by-side
- Analyze detailed metrics for individual experiments
- Visualize accuracy, operational, and error metrics
- Explore mismatches and document-level details

Usage:
    streamlit run dashboard.py
"""

import base64
import datetime
import json
from logging import ERROR
import os
from contextlib import suppress
from typing import Any, Tuple

import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import streamlit as st  # type: ignore
from dvc.repo import Repo  # type: ignore
from git import Repo as GitRepo  # type: ignore
from pandas.io.formats import style  # type: ignore
from config import (  # type: ignore
    ACCURACY_METRICS,
    COST_METRICS,
    ERROR_METRICS,
    KEY_METRICS,
    LATENCY_METRICS,
    OPERATIONAL_METRICS,
    DETAILS_CONFIG,
)
from utils import (  # type: ignore
    apply_enhanced_table_formatting,
    create_cost_ranking,
    create_performance_ranking,
    create_time_ranking,
    extract_combined_configuration_matrix,
    extract_configuration_matrix,
    extract_llm_requests_data,
    format_column_headers,
    format_metric_value,
    generate_experiment_colors,
    load_experiment_summary,
)

from delete.commons.utils import configure_logger

# Create logger for this module
LOGGER = configure_logger(__name__, log_to_stdout=True)

dvc_repo = Repo(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Initialize Git repository for metadata extraction
git_repo = GitRepo(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Configure Streamlit page
st.set_page_config(
    page_title=f"OmniDocBench Experiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


class ExperimentLoader:
    """Handles loading and caching of experiment data."""

    def __init__(self) -> None:
        self.cached_experiments: dict[str, dict[str, Any]] = {}

    def load_experiment_data(self, experiment_name: str, experiment_commit: str) -> dict[str, Any] | None:
        """Load experiment data from the experiment directory."""
        if experiment_name in self.cached_experiments:
            return self.cached_experiments[experiment_name]

        # Use the utils function to load experiment data
        data = load_experiment_summary(experiment_name, experiment_commit)

        if data is None:
            return None

        # Cache the data
        self.cached_experiments[experiment_name] = data

        return data if data else None

    @staticmethod
    @st.cache_data(ttl=900, show_spinner="Loading experiment metadata...", show_time=True)  # Cache for 15 minutes
    def _load_metadata_cached(experiment_names: tuple, commits: tuple) -> dict[str, dict[str, Any]]:
        """Cached metadata loading to avoid reloading on filter changes."""
        metadata: dict[str, dict[str, Any]] = {}

        def load_single_metadata(exp_name: str, commit: str) -> Tuple[str, dict[str, Any]]:
            """Load minimal metadata for a single experiment using Git information."""
            try:
                # Get commit information from Git
                git_commit = git_repo.commit(commit)
                author_name = git_commit.author.name
                author_email = git_commit.author.email
                user = f"{author_name} <{author_email}>"
                timestamp = int(git_commit.committed_date)

                # Determine the branches that contain this commit
                branches_output = git_repo.git.branch("--contains", commit, "--all")
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

                return exp_name, {
                    "user": user,
                    "git_branch": git_branch,
                    "commit": commit,
                    "timestamp": timestamp,
                    "loaded_successfully": True,
                }

            except Exception as e:
                # Handle any Git error (commit not found, branch detection failed, etc.)
                return exp_name, {
                    "user": "Unknown",
                    "git_branch": "Unknown",
                    "commit": commit,
                    "timestamp": 0,
                    "loaded_successfully": False,
                    "error": f"Git error: {str(e)}",
                }

        # Track duplicates for debugging
        seen_experiments = set()
        duplicates = []

        for exp_name, commit in zip(experiment_names, commits, strict=False):
            if exp_name in seen_experiments:
                duplicates.append(exp_name)
                LOGGER.warning(f"Duplicate experiment name found: {exp_name}")
            seen_experiments.add(exp_name)

            exp_name, meta = load_single_metadata(exp_name, commit)

            # Create unique key if duplicate exists
            unique_key = exp_name
            counter = 1
            while unique_key in metadata:
                unique_key = f"{exp_name}_{counter}"
                counter += 1

            metadata[unique_key] = meta

        return metadata

    def load_metadata(self, experiment_names: list[str], commits: list[str]) -> dict[str, dict[str, Any]]:
        """Load experiment metadata in parallel with minimal data."""
        # Convert to tuples for caching (lists are not hashable)
        exp_names_tuple = tuple(experiment_names)
        commits_tuple = tuple(commits)

        # Use cached version - this will only reload if the experiment list changes
        metadata: dict[str, dict[str, Any]] = self._load_metadata_cached(exp_names_tuple, commits_tuple)

        # Calculate summary statistics for display
        successful_loads = sum(1 for meta in metadata.values() if meta.get("loaded_successfully", False))
        failed_loads = len(metadata) - successful_loads

        # Only show warning for actual failures
        if failed_loads > 0:
            st.sidebar.warning(f"⚠️ Failed to load metadata for {failed_loads} experiments")

        # Show detailed info about loaded metadata
        if st.sidebar.checkbox("🔍 Show Detailed Info", value=False):
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔍 Detailed Info")

            # Loading Status with visual indicators (as part of detailed info)
            st.sidebar.markdown("#### 📊 Loading Status")

            # Create columns for metrics
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("✅ Loaded", successful_loads, help="Successfully loaded experiments")
            with col2:
                if failed_loads > 0:
                    st.metric("❌ Failed", failed_loads, help="Failed to load experiments")
                else:
                    st.metric("❌ Failed", "0", help="No failed experiments")

            # Users section with improved formatting (always collapsed)
            unique_users = {meta["user"] for meta in metadata.values() if meta["user"] != "Unknown"}
            st.sidebar.markdown("#### 👥 Users")

            if unique_users:
                st.sidebar.markdown(f"**{len(unique_users)} unique users found:**")
                # Always keep collapsed for cleaner interface
                with st.sidebar.expander("View user list", expanded=False):
                    for user in sorted(unique_users):
                        # Extract name and email for single-line display
                        display_name = user.split(" <")[0] if " <" in user else user
                        email = user.split("<")[1].replace(">", "") if "<" in user else ""
                        if email:
                            st.markdown(f"• **{display_name}** `<{email}>`")
                        else:
                            st.markdown(f"• **{display_name}**")
            else:
                st.sidebar.info("No user information available")

            # Branches section with improved formatting (always collapsed)
            unique_branches = {meta["git_branch"] for meta in metadata.values() if meta["git_branch"] != "Unknown"}
            st.sidebar.markdown("#### 🌿 Git Branches")

            if unique_branches:
                st.sidebar.markdown(f"**{len(unique_branches)} unique branches found:**")
                # Always keep collapsed for cleaner interface
                with st.sidebar.expander("View branch list", expanded=False):
                    for branch in sorted(unique_branches):
                        st.markdown(f"• `{branch}`")
            else:
                st.sidebar.info("No branch information available")

            # Error summary with better formatting (only if there are errors)
            if failed_loads > 0:
                st.sidebar.markdown("#### ⚠️ Error Details")
                error_types: dict[str, int] = {}
                for meta in metadata.values():
                    if not meta.get("loaded_successfully", False):
                        error = meta.get("error", "Unknown error")
                        error_types[error] = error_types.get(error, 0) + 1

                with st.sidebar.expander(f"View {len(error_types)} error type(s)", expanded=failed_loads <= 5):
                    for error, count in sorted(error_types.items()):
                        # Truncate long error messages for readability
                        display_error = error[:50] + "..." if len(error) > 50 else error
                        st.markdown(f"• **{count}x** {display_error}")

            st.sidebar.markdown("---")

        return metadata

    @staticmethod
    def get_all_experiments() -> Tuple[dict[str, list[Tuple[str, str]]], list[str]]:
        """
        Modern approach to get all experiments using DVC API.

        Returns:
            Tuple of (experiments_per_commit, all_commit_shas)
        """
        try:
            # Use the existing method as primary, but enhance with exp_show data
            experiments_per_commit = dvc_repo.experiments.ls(all_commits=True)

            # Get more detailed experiment data using exp_show (if available)
            with suppress(Exception):
                # This provides more detailed information for potential future enhancements
                _ = dvc_repo.experiments.show()

            # Extract commit SHAs and sort by recency if possible
            all_commit_shas = list(experiments_per_commit.keys())

            # Sort commits by Git commit date
            try:

                def get_commit_date(commit_sha: str) -> int:
                    try:
                        return git_repo.commit(commit_sha).committed_date
                    except Exception:
                        return 0

                all_commit_shas.sort(key=get_commit_date, reverse=True)
            except Exception:
                # Fallback to string sort if Git operations fail
                all_commit_shas.sort(reverse=True)

            return experiments_per_commit, all_commit_shas

        except Exception as e:
            st.error(f"Failed to load experiments: {e}")
            return {}, []


class MetricsAnalyzer:
    """Handles analysis and visualization of metrics data."""

    @staticmethod
    def extract_main_metrics(data: dict[str, Any]) -> dict[str, float]:
        """Extract the main comparison metrics from experiment data."""
        metrics = {}

        processed_metrics = data.get("summary", {}).get("processed", {})
        end_to_end_metrics = data.get("summary", {}).get("end_to_end", {})
        summary_metrics = data.get("summary", {})
        error_info_metrics = data.get("error_info", {})
        custom_metrics = data.get("metrics", {})

        # Extract key metrics with best values
        key_metrics = KEY_METRICS

        for metric in key_metrics:
            if metric in processed_metrics:
                metrics["processed." + metric] = processed_metrics[metric]
            if metric in end_to_end_metrics:
                metrics["end_to_end." + metric] = end_to_end_metrics[metric]
            if metric in summary_metrics:
                metrics[metric] = summary_metrics[metric]
            if metric in error_info_metrics:
                metrics[metric] = error_info_metrics[metric]
            if metric in custom_metrics:
                metrics["metrics." + metric] = custom_metrics[metric]

        return metrics

    @staticmethod
    def extract_operational_metrics(data: dict[str, Any]) -> dict[str, float]:
        """Extract operational metrics from experiment data."""
        metrics = {}

        # Extract operational metrics from summary
        operational_metrics = OPERATIONAL_METRICS

        for metric in operational_metrics:
            if metric in data.get("summary", {}):
                metrics["operations." + metric] = data["summary"][metric]
            elif metric in data.get("operations", {}):
                metrics["operations." + metric] = data["operations"][metric]
            elif metric in data.get("metrics", {}):
                metrics["operations." + metric] = data["metrics"][metric]

        return metrics

    @staticmethod
    def create_comparison_dataframe(
        experiments: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Create a comparison dataframe from multiple experiments."""
        comparison_data = []

        for exp_path, exp_data in experiments.items():
            exp_name = os.path.basename(exp_path)
            row: dict[str, Any] = {"Experiment": exp_name}

            # Add main metrics
            main_metrics = MetricsAnalyzer.extract_main_metrics(exp_data)
            row.update(main_metrics)

            # Add operational metrics
            operational_metrics = MetricsAnalyzer.extract_operational_metrics(exp_data)
            row.update(operational_metrics)

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    @staticmethod
    def create_metrics_bar_chart(
        df: pd.DataFrame,
        metrics: list[str],
        title: str,
        experiment_colors: dict[str, str] | None,
    ) -> None | go.Figure:
        """Create a bar chart for comparing metrics across experiments with consistent coloring and visual grouping."""
        if df.empty or not metrics:
            return None

        fig = go.Figure()

        # Detect metric groups for visual separation
        def get_metric_group(metric: str) -> str:
            if metric.startswith("end_to_end"):
                return "End-to-End"
            if metric.startswith("processed"):
                return "Processed"
            if metric.startswith("operations"):
                return "Operations"
            if metric.startswith("metrics"):
                return "Custom"
            return "Other"

        def _iterative_search(row: pd.Series, metric: str) -> Any | None:
            """Search for metric in nested dictionaries within the row."""
            parts = metric.split(".")
            current_value: Any = row
            for part in parts:
                if isinstance(current_value, dict) and part in current_value:
                    current_value = current_value[part]
                else:
                    return None
            return current_value

        # Group metrics by category
        grouped_metrics: dict[str, Any] = {}
        for metric in metrics:
            group = get_metric_group(metric)
            if group not in grouped_metrics:
                grouped_metrics[group] = []
            grouped_metrics[group].append(metric)

        # Create display names with grouping
        display_names = []
        display_values_by_experiment: dict[str, Any] = {}

        for _, row in df.iterrows():
            experiment_name = os.path.basename(row["Experiment"])
            display_values_by_experiment[experiment_name] = []

        # Build grouped display names and collect values
        for group_name, group_metrics in grouped_metrics.items():
            for i, metric in enumerate(group_metrics):
                # Clean up metric names for display
                clean_name = (
                    metric.replace("operations", "").replace("seconds", "").replace("_", " ").replace(".", " ").title()
                )

                # Add group prefix for first metric in each group for clarity
                display_name = f"{group_name}\n{clean_name}" if i == 0 and len(grouped_metrics) > 1 else clean_name

                display_names.append(display_name)

                # Collect values for each experiment
                for _, row in df.iterrows():
                    experiment_name = os.path.basename(row["Experiment"])
                    if metric in df.columns:
                        display_values_by_experiment[experiment_name].append(row[metric])
                    elif _iterative_search(row, metric) is not None:
                        display_values_by_experiment[experiment_name].append(_iterative_search(row, metric))
                    else:
                        display_values_by_experiment[experiment_name].append(0)

        # Create traces for each experiment
        for _, row in df.iterrows():
            experiment_name = os.path.basename(row["Experiment"])

            # Get consistent color for this experiment
            color = "#1f77b4"  # Default blue
            if experiment_colors and experiment_name in experiment_colors:
                color = experiment_colors[experiment_name]

            values = display_values_by_experiment[experiment_name]

            fig.add_trace(
                go.Bar(
                    name=experiment_name,
                    x=display_names,
                    y=values,
                    text=[
                        (format_metric_value(val, metrics[i]) if isinstance(val, (int, float)) else str(val))
                        for i, val in enumerate(values)
                    ],
                    textposition="auto",
                    marker_color=color,
                    opacity=0.8,
                )
            )

        # Add visual separators between groups if there are multiple groups
        if len(grouped_metrics) > 1:
            # Calculate separator positions
            separator_positions = []
            current_pos = 0
            for group_metrics in list(grouped_metrics.values())[:-1]:  # All but last group
                current_pos += len(group_metrics)
                separator_positions.append(current_pos - 0.5)

            # Add vertical lines as separators
            for pos in separator_positions:
                fig.add_vline(
                    x=pos,
                    line={"color": "rgba(128,128,128,0.3)", "width": 2, "dash": "dash"},
                    annotation_text="",
                )

        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Metric Value",
            barmode="group",
            height=500,
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            xaxis={
                "tickangle": 45,  # Rotate labels for better readability
                "tickfont": {"size": 10},
            },
        )

        return fig


class DashboardUI:
    """Handles the Streamlit UI components."""

    def __init__(self) -> None:
        self.loader = ExperimentLoader()
        self.analyzer = MetricsAnalyzer()

    @staticmethod
    def safe_json_display(data: Any) -> str:
        """
        Convert data to a JSON-safe string format for display in Streamlit.
        Handles boolean and null values that cause issues with st.json().

        Args:
            data: The data to convert

        Returns:
            JSON string safe for display
        """

        def convert_for_display(obj: Any) -> Any:
            """Recursively convert problematic values to strings."""
            if isinstance(obj, dict):
                return {key: convert_for_display(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [convert_for_display(item) for item in obj]
            if isinstance(obj, bool):
                return str(obj).lower()  # Convert True/False to "true"/"false"
            if obj is None:
                return "null"
            return obj

        try:
            converted_data = convert_for_display(data)
            return json.dumps(converted_data, indent=2)
        except Exception:
            return str(data)

    def render_sidebar(self) -> Tuple[list[Tuple[str, str]], list[str], list[str], dict[str, int]]:
        """Render the sidebar with experiment selection."""

        def format_experiment_name(exp_name: str) -> str:
            """Format experiment name with metadata if available."""
            base_text = f"{'✓ ' if exp_name in filtered_experiments else '  '}{exp_name}"
            if exp_name in experiment_metadata:
                meta = experiment_metadata[exp_name]
                if meta.get("loaded_successfully", False):
                    user = meta["user"]
                    branch = meta["git_branch"]
                    if user != "Unknown" or branch != "Unknown":
                        base_text += f" ({user}@{branch})"
            return base_text

        # Display company logo based on theme
        def display_company_logo() -> None:
            """Display the appropriate company logo based on the current theme."""
            # Define logo paths
            logos_dir = os.path.join(os.path.dirname(__file__), "logos")
            dark_logo_path = os.path.join(logos_dir, "Theme=Dark, Scheme=color.svg")
            light_logo_path = os.path.join(logos_dir, "Theme=Light, Scheme=color.svg")

            # Read and encode SVG files as base64
            with open(light_logo_path, "rb") as f:
                light_svg_b64 = base64.b64encode(f.read()).decode()

            with open(dark_logo_path, "rb") as f:
                dark_svg_b64 = base64.b64encode(f.read()).decode()

            # Create responsive logo display using CSS and base64 encoded SVGs
            logo_html = f"""
            <div style="display: flex; justify-content: center; margin: 10px 0 20px 0;">
                <div class="logo-container">
                    <img class="logo-light" src="data:image/svg+xml;base64,{light_svg_b64}" 
                            style="max-width: 200px; height: auto;" alt="Company Logo">
                    <img class="logo-dark" src="data:image/svg+xml;base64,{dark_svg_b64}" 
                            style="max-width: 200px; height: auto; display: none;" alt="Company Logo">
                </div>
            </div>
            <style>
                @media (prefers-color-scheme: dark) {{
                    .logo-light {{ display: none !important; }}
                    .logo-dark {{ display: block !important; }}
                }}
                @media (prefers-color-scheme: light) {{
                    .logo-light {{ display: block !important; }}
                    .logo-dark {{ display: none !important; }}
                }}
            </style>
            """

            st.sidebar.markdown(logo_html, unsafe_allow_html=True)

        display_company_logo()

        # Base directory selection
        st.sidebar.subheader("🗂️ Experiment Base Commit")

        # Use modernized experiment discovery
        experiments_per_commit, commits = ExperimentLoader.get_all_experiments()

        origin_experiments = dvc_repo.experiments.ls(all_commits=True, git_remote="origin")
        not_pulled_experiments = [
            exp[0]
            for c in origin_experiments
            for exp in origin_experiments[c]
            if exp[0] not in [c[0] for _, exps in experiments_per_commit.items() for c in exps]
        ]
        if len(not_pulled_experiments) > 0:
            st.sidebar.warning(
                "Some experiments are not pulled from origin. Please run `pdm run exp-pull --all` to sync."
            )
            st.sidebar.write(
                "Not pulled experiments: "
                + ", ".join(not_pulled_experiments[:5])
                + ("..." if len(not_pulled_experiments) > 5 else "")
            )
        selected_commits = st.sidebar.multiselect(
            "Select base commits for experiments:",
            options=commits,
            default=commits if commits else [],
            help="Choose the base commit to load experiments from",
        )

        # Find all experiments from selected commits
        all_experiment_dirs = [exp[0] for c in selected_commits for exp in experiments_per_commit.get(c, "")]

        # Find time for experiments

        if not all_experiment_dirs:
            st.sidebar.warning("No experiments found for the specified commits.")
            return [], [], [], {"total_experiments": 0, "filtered_experiments": 0}

        # Load metadata from all experiments to enable filtering
        st.sidebar.subheader("🏷️ Metadata Filters")

        # Find corresponding commits for experiments
        exp_to_commit = {}
        for commit in selected_commits:
            for exp, _ in experiments_per_commit.get(commit, []):
                exp_to_commit[exp] = commit

        # Load metadata efficiently using cached processing
        experiment_metadata = self.loader.load_metadata(
            all_experiment_dirs, [exp_to_commit[exp] for exp in all_experiment_dirs]
        )

        # Extract filter options from loaded metadata
        # Get both known and unknown users/branches
        known_users = sorted(
            {
                meta["user"]
                for meta in experiment_metadata.values()
                if meta.get("loaded_successfully", False) and meta["user"] != "Unknown"
            }
        )
        known_branches = sorted(
            {
                meta["git_branch"]
                for meta in experiment_metadata.values()
                if meta.get("loaded_successfully", False) and meta["git_branch"] != "Unknown"
            }
        )

        # User filter
        selected_users = st.sidebar.multiselect(
            "Filter by User:",
            options=known_users + ["Unknown"],
            default=[],
            help="Select users to filter experiments. Leave empty to include all users.",
        )

        # Git branch filter
        selected_branches = st.sidebar.multiselect(
            "Filter by Git Branch:",
            options=known_branches + ["Unknown"],
            default=[],
            help="Select git branches to filter experiments. Leave empty to include all branches.",
        )

        # Filter experiments based on metadata
        filtered_experiments = []
        for exp_name, metadata in experiment_metadata.items():
            # Always include experiments that failed to load metadata (to avoid losing them)
            if not metadata.get("loaded_successfully", False):
                if not selected_users and not selected_branches:
                    filtered_experiments.append(exp_name)
                continue

            user_match = not selected_users or metadata["user"] in selected_users
            branch_match = not selected_branches or metadata["git_branch"] in selected_branches

            if user_match and branch_match:
                filtered_experiments.append(exp_name)

        # Experiment selection - show ALL experiments from selected commits
        st.sidebar.subheader("🔬 Select Experiments")

        # Display filter summary
        if selected_users or selected_branches:
            # Option to select all filtered experiments
            select_all_filtered = st.sidebar.checkbox(
                "Auto-select Filtered Experiments",
                value=False,
                help="Automatically select all experiments that match the current filters",
            )

            if select_all_filtered:
                selected_experiments = filtered_experiments
            else:
                filtered_experiments.sort(key=lambda x: experiment_metadata[x]["timestamp"], reverse=True)
                selected_experiments = st.sidebar.multiselect(
                    "Choose experiments to analyze:",
                    options=all_experiment_dirs,
                    default=(filtered_experiments if len(filtered_experiments) <= 3 else []),
                    help="Select experiments to analyze. All experiments from selected commits are available.",
                    format_func=format_experiment_name,
                )
        else:
            st.sidebar.write(f"*{len(all_experiment_dirs)} experiments available*")
            all_experiment_dirs.sort(key=lambda x: experiment_metadata[x]["timestamp"], reverse=True)
            formatted_experiment_dirs = []
            for exp in all_experiment_dirs:
                timestamp_str = datetime.datetime.fromtimestamp(experiment_metadata[exp]["timestamp"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                formatted_experiment_dirs.append(f"{exp} ({timestamp_str})")
            selected_experiments = st.sidebar.multiselect(
                "Choose experiments to analyze:",
                options=formatted_experiment_dirs,
                default=formatted_experiment_dirs if len(formatted_experiment_dirs) <= 3 else [],
                help="Select experiments to analyze from selected commits",
                format_func=format_experiment_name,
            )
            selected_experiments = [exp.split(" (")[0] for exp in selected_experiments]

        st.sidebar.info(
            """✅ **Creating Experiments**:
```bash
pdm run exp-run [--name <experiment_name> -m <message>]
```

⏲️ **Queuing Experiments**:
```bash
pdm run exp-run --queue [--name <experiment_name> -m <message>]
pdm run exp-start-queue # Start the queue
pdm run exp-show-queue # Check the queue status
pdm run exp-stop-queue # Stop the queue
```

♻️ **Applying Experiments**:
```bash
pdm run exp-checkout <experiment_name>
```

📤 **Sharing Experiments**:
```bash
pdm run exp-push <experiment_name>
```

❌ **Deleting Experiments**:
```bash
pdm run exp-clean-remote <experiment_name> # This will delete the pushed experiment.
pdm run exp-clean-local <experiment_name> # This will delete the experiment from the local repository.
```"""
        )

        # Return experiments with their commits and filter metadata
        selected_experiments_with_commits = [
            (exp, experiment_metadata[exp]["commit"]) for exp in selected_experiments if exp in experiment_metadata
        ]

        filter_metadata = {
            "total_experiments": len(all_experiment_dirs),
            "filtered_experiments": len(filtered_experiments)
            if (selected_users or selected_branches)
            else len(all_experiment_dirs),
        }

        return selected_experiments_with_commits, selected_users, selected_branches, filter_metadata

    @staticmethod
    @st.cache_data(ttl=900, show_spinner="Generating comparison data...", show_time=True)  # Cache for 15 minutes
    def _generate_comparison_data_cached(experiment_paths: tuple, experiment_data: tuple) -> dict[str, Any]:
        """Cached comparison data generation to avoid reprocessing on filter changes."""
        # Convert tuples back to dictionaries for processing
        experiments = dict(zip(experiment_paths, experiment_data, strict=False))

        comparison_data = {}

        try:
            # Generate configuration matrix
            if len(experiments) == 1:
                # Single experiment - use original logic
                exp_name = list(experiments.keys())[0]
                exp_data = experiments[exp_name]
                config_matrix = extract_configuration_matrix(exp_name, exp_data)
            else:
                # Multiple experiments - use combined configuration matrix
                config_matrix = extract_combined_configuration_matrix(experiments)

            comparison_data["config_matrix"] = config_matrix

            # Generate rankings for multiple experiments
            if len(experiments) > 1:
                comparison_data["performance_ranking"] = create_performance_ranking(experiments)
                comparison_data["cost_ranking"] = create_cost_ranking(experiments)
                comparison_data["time_ranking"] = create_time_ranking(experiments)

            # Generate experiment colors
            comparison_data["experiment_colors"] = generate_experiment_colors(experiments)

            # Create comparison dataframe for charts
            analyzer = MetricsAnalyzer()
            comparison_data["comparison_df"] = analyzer.create_comparison_dataframe(experiments)

            comparison_data["success"] = True

        except Exception as e:
            comparison_data["success"] = False
            comparison_data["error"] = str(e)

        return comparison_data

    def render_main_comparison(self, experiments: dict[str, dict[str, Any]]) -> None:
        """Render the main comparison view."""
        st.title("🔬 Experiment Metrics Comparison")

        if len(experiments) == 0:
            st.info("Select experiments from the sidebar to begin analysis.")
            return

        # Prepare data for caching (convert to tuples for hashability)
        experiment_paths = tuple(experiments.keys())
        experiment_data = tuple(experiments.values())

        # Generate comparison data using cached function
        comparison_data = self._generate_comparison_data_cached(experiment_paths, experiment_data)

        if not comparison_data.get("success", False):
            st.error(f"Failed to generate comparison data: {comparison_data.get('error', 'Unknown error')}")
            return

        # Extract cached results
        config_matrix = comparison_data["config_matrix"]
        experiment_colors = comparison_data["experiment_colors"]
        comparison_df = comparison_data["comparison_df"]

        # Configuration Matrix for all experiments
        st.subheader("📋 Configuration Matrix")

        # Add explanation for color coding when multiple experiments are selected
        if len(experiments) > 1:
            st.info(
                "💡 **Visual Correlation**: Each experiment has a consistent row color across all tables below for "
                "easy comparison."
            )

        try:
            if not config_matrix.empty:
                if len(experiments) > 1:
                    # Apply enhanced formatting with row coloring for multiple experiments
                    styled_config_matrix = apply_enhanced_table_formatting(
                        config_matrix,
                        experiment_colors,
                        "Experiment",
                        format_headers=True,
                    )
                    st.dataframe(styled_config_matrix, width="stretch", hide_index=True)
                else:
                    # Single experiment - apply basic formatting without coloring
                    formatted_matrix = format_column_headers(config_matrix)
                    st.dataframe(formatted_matrix, width="stretch", hide_index=True)
            else:
                st.error("No configuration matrix available.")
        except Exception as e:
            st.error(f"Could not load configuration matrix: {e}")

        # Rankings (only for multiple experiments)
        if len(experiments) > 1:
            # Performance Ranking (Accuracy metrics)
            st.subheader("🏆 Performance Ranking")
            try:
                ranking_df = comparison_data.get("performance_ranking")
                if ranking_df is not None and not ranking_df.empty:
                    styled_ranking_df = self.display_formatted_ranking_table(
                        ranking_df, experiment_colors, "experiment"
                    )
                    if styled_ranking_df is not None:
                        st.dataframe(styled_ranking_df, width="stretch", hide_index=True)
                else:
                    st.error("Performance ranking data not available.")
            except Exception as e:
                st.error(f"Could not generate performance ranking: {e}")

            # Cost Ranking
            st.subheader("💰 Cost Ranking")
            try:
                cost_ranking_df = comparison_data.get("cost_ranking")
                if cost_ranking_df is not None and not cost_ranking_df.empty:
                    styled_cost_ranking_df = self.display_formatted_ranking_table(
                        cost_ranking_df, experiment_colors, "experiment"
                    )
                    if styled_cost_ranking_df is not None:
                        st.dataframe(
                            styled_cost_ranking_df,
                            width="stretch",
                            hide_index=True,
                        )
                else:
                    st.error("Cost ranking data not available.")
            except Exception as e:
                st.warning(f"Could not generate cost ranking: {e}")

            # Time Ranking
            st.subheader("⏱️ Time Ranking")
            try:
                time_ranking_df = comparison_data.get("time_ranking")
                if time_ranking_df is not None and not time_ranking_df.empty:
                    styled_time_ranking_df = self.display_formatted_ranking_table(
                        time_ranking_df, experiment_colors, "experiment"
                    )
                    if styled_time_ranking_df is not None:
                        st.dataframe(
                            styled_time_ranking_df,
                            width="stretch",
                            hide_index=True,
                        )
                else:
                    st.error("Time ranking data not available.")
            except Exception as e:
                st.warning(f"Could not generate time ranking: {e}")

        if comparison_df.empty:
            st.warning("No comparable metrics found in the selected experiments.")
            return

        # Main metrics comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Accuracy Metrics")
            accuracy_metrics = ACCURACY_METRICS

            accuracy_fig = self.analyzer.create_metrics_bar_chart(
                comparison_df,
                accuracy_metrics,
                "Accuracy Metrics Comparison",
                experiment_colors,
            )
            if accuracy_fig:
                st.plotly_chart(accuracy_fig, width="stretch")

        with col2:
            st.subheader("⚠️ Error Metrics")
            error_metrics = ERROR_METRICS

            error_fig = self.analyzer.create_metrics_bar_chart(
                comparison_df,
                error_metrics,
                "Error Metrics Comparison",
                experiment_colors,
            )
            if error_fig:
                st.plotly_chart(error_fig, width="stretch")

        # Operational metrics - separated into two focused charts
        st.subheader("⚙️ Operational Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**💰 Cost Metrics**")
            cost_metrics = COST_METRICS

            cost_fig = self.analyzer.create_metrics_bar_chart(
                comparison_df,
                cost_metrics,
                "Cost Metrics Comparison",
                experiment_colors,
            )
            if cost_fig:
                st.plotly_chart(cost_fig, width="stretch")

        with col2:
            st.write("**⏱️ Latency Metrics**")
            latency_metrics = LATENCY_METRICS

            latency_fig = self.analyzer.create_metrics_bar_chart(
                comparison_df,
                latency_metrics,
                "Latency Metrics Comparison",
                experiment_colors,
            )
            if latency_fig:
                st.plotly_chart(latency_fig, width="stretch")

    def render_experiment_details(self, experiments: dict[str, dict[str, Any]]) -> None:
        """Render detailed analysis for individual experiments."""
        st.title("📋 Detailed Experiment Analysis")

        if len(experiments) == 0:
            st.info("Select experiments from the sidebar to view details.")
            return

        # Experiment selection for detailed view
        experiment_names = list(experiments.keys())
        selected_experiment = st.selectbox(
            "Select experiment for detailed analysis:",
            options=experiment_names,
            format_func=lambda x: os.path.basename(x),
        )

        if not selected_experiment:
            return

        experiment_data = experiments[selected_experiment]

        # Create tabs for different analysis sections
        tab1, tab2 = st.tabs(
            [
                "📊 Metrics Overview",
                "🤖 LLM Calls Analysis",
            ]
        )

        with tab1:
            self.render_metrics_overview(experiment_data)

        with tab2:
            self.render_llm_calls_analysis(experiment_data)

    def render_metrics_overview(self, config_data: dict[str, Any]) -> None:
        """Render metrics overview for a configuration."""
        st.subheader("📊 Metrics Overview")

        # Key performance indicators
        accuracy_metrics = config_data.get("summary", {}).get("processed", {})
        end_to_end_metrics = config_data.get("summary", {}).get("end_to_end", {})
        error_info = config_data.get("error_info", {})
        summary_metrics = config_data.get("summary", {})
        custom_metrics = config_data.get("metrics", {})

        details_config = DETAILS_CONFIG["overview"]

        for section, metrics in details_config.items():
            with st.container(border=True):
                st.markdown(f"**{section}**")
                cols = st.columns(len(metrics))
                for i, (title, metric_config) in enumerate(metrics.items()):
                    metric_values = [
                        (
                            accuracy_metrics.get(f)
                            or end_to_end_metrics.get(f)
                            or summary_metrics.get(f)
                            or error_info.get(f)
                            or custom_metrics.get(f)
                            or 0
                        )
                        for f in metric_config["fields"]
                    ]
                    metric_config["format"](cols[i], title, metric_values, metric_config["fields"])

        # Processing statistics
        st.subheader("📈 Processing Statistics")

        processing_config = DETAILS_CONFIG["processing_stats"]

        cols = st.columns(len(processing_config))
        for i, (title, metric_config) in enumerate(processing_config.items()):
            metric_values = [
                (
                    accuracy_metrics.get(f)
                    or end_to_end_metrics.get(f)
                    or summary_metrics.get(f)
                    or error_info.get(f)
                    or custom_metrics.get(f)
                    or 0
                )
                for f in metric_config["fields"]
            ]
            metric_config["format"](cols[i], title, metric_values, metric_config["fields"])

    def render_llm_calls_analysis(self, experiment_data: dict[str, Any]) -> None:
        """Render LLM calls analysis."""
        st.subheader("🤖 LLM Calls Analysis")

        # Extract LLM requests data
        docs_data = extract_llm_requests_data(experiment_data)

        if not docs_data:
            st.error("No LLM requests data available for this experiment.")
            return

        # Display available documents
        st.info(f"📄 **Total Documents:** {len(docs_data)}")

        # Document selector
        document_names = list(docs_data.keys())
        selected_document = st.selectbox(
            "Select document to analyze:",
            options=document_names,
            help="Choose a document to view its LLM processing nodes",
        )

        if selected_document and selected_document in docs_data:
            doc_data = docs_data[selected_document]

            # Extract node names from the document data
            node_names = list(doc_data.keys())

            if not node_names:
                st.error("No processing nodes found for this document.")
                return

            st.info(f"🔗 **Processing Nodes in Document:** {len(node_names)}")

            # Node selector
            selected_node = st.selectbox(
                "Select processing node to analyze:",
                options=node_names,
                help="Choose a processing node to view its messages, schema, and configuration",
            )

            if selected_node and selected_node in doc_data:
                node_data = doc_data[selected_node]

                # Messages Analysis
                st.subheader("📝 LLM Messages")

                messages = node_data.get("messages", [])
                if messages:
                    for i, message_exchange in enumerate(messages):
                        with st.expander("💬 Messages", expanded=i == 0):
                            for _, message in enumerate(message_exchange):
                                role = message.get("role", "")
                                content = message.get("content", "")

                                if role == "system":
                                    with st.expander("🖥️ System Message", expanded=False):
                                        if isinstance(content, str):
                                            st.markdown(content, unsafe_allow_html=True)
                                        elif isinstance(content, list):
                                            text_parts = []
                                            for content_item in content:
                                                if (
                                                    isinstance(content_item, dict)
                                                    and content_item.get("type") == "text"
                                                ):
                                                    text_parts.append(content_item.get("text", ""))
                                            if text_parts:
                                                combined_text = "\n\n".join(text_parts)
                                                st.markdown(combined_text, unsafe_allow_html=True)
                                        else:
                                            st.warning("No system message in valid format <str> content found")

                                elif role == "user":
                                    with st.expander("👤 User Message", expanded=False):
                                        if isinstance(content, list):
                                            # Handle multi-modal content (text + images)
                                            text_parts = []
                                            for content_item in content:
                                                if isinstance(content_item, dict):
                                                    if content_item.get("type") == "text":
                                                        text_parts.append(content_item.get("text", ""))
                                                    elif content_item.get("type") == "image_url":
                                                        # Append image URL as plain text
                                                        image_url = content_item.get("image_url", {}).get("url", "")
                                                        if image_url:
                                                            text_parts.append(f"{image_url}")

                                            if text_parts:
                                                combined_text = "\n\n".join(text_parts)
                                                st.markdown(combined_text, unsafe_allow_html=True)
                                        else:
                                            st.warning("No user message in valid format <list> content found")

                                elif role == "assistant":
                                    with st.expander("🤖 Assistant Message", expanded=False):
                                        if isinstance(content, list):
                                            # Handle list content for assistant messages
                                            text_parts = []
                                            for content_item in content:
                                                if (
                                                    isinstance(content_item, dict)
                                                    and content_item.get("type") == "text"
                                                ):
                                                    text_parts.append(content_item.get("text", ""))

                                            if text_parts:
                                                combined_text = "\n\n".join(text_parts)
                                                st.markdown(combined_text, unsafe_allow_html=True)
                                        else:
                                            st.warning("No assistant message in valid format <list> content found")
                else:
                    st.warning("No messages found for this node")

                # Configuration Analysis
                st.subheader("⚙️ Node Configuration")

                config = node_data.get("config", {})
                if config:
                    # Display full config in expandable section
                    with st.expander("📄 Full Configuration", expanded=False):
                        config_str = DashboardUI.safe_json_display(config)
                        st.code(config_str, language="json")
                else:
                    st.warning("No configuration found for this node")

                # Schema Analysis
                st.subheader("📑 Response Schema")

                schema = node_data.get("schema", {})
                if schema:
                    with st.expander("📂 Expected Response Schema", expanded=False):
                        # Display schema as formatted JSON
                        schema_str = DashboardUI.safe_json_display(schema)
                        st.code(schema_str, language="json")
                else:
                    st.warning("No schema found for this node")

    def display_formatted_ranking_table(
        self,
        ranking_df: pd.DataFrame,
        experiment_colors: dict[str, str],
        experiment_column: str = "experiment",
    ) -> None | pd.DataFrame | style.Styler:
        """Display a ranking table with formatted values and consistent coloring."""
        if ranking_df.empty:
            return None

        # Create a display dataframe with formatted values
        display_df = ranking_df.copy()

        # Replace raw metric columns with formatted versions where available
        for col in display_df.columns:
            formatted_col = col + "_formatted"
            if formatted_col in display_df.columns:
                display_df[col] = display_df[formatted_col]
                display_df.drop(columns=[formatted_col], inplace=True)

        # Remove experiment_path column if it exists (used for internal purposes)
        if "experiment_path" in display_df.columns:
            display_df.drop(columns=["experiment_path"], inplace=True)

        # Apply enhanced formatting including styling, coloring, and headers
        return apply_enhanced_table_formatting(display_df, experiment_colors, experiment_column, format_headers=True)

    def run(self) -> None:
        """Main application runner."""
        # Render sidebar and get filtered experiments
        selected_experiments, selected_users, selected_branches, filter_metadata = self.render_sidebar()

        # Load experiment data for selected experiments
        experiments = {}
        for exp_name, exp_commit in selected_experiments:
            exp_data = self.loader.load_experiment_data(exp_name, exp_commit)
            if exp_data:
                experiments[exp_name] = exp_data

        # Main content area
        if experiments:
            # Show filter summary if filters are applied
            if selected_users or selected_branches:
                # Create an expandable section for Active Filters in main content
                with st.expander("� **Active Filters & Results**", expanded=True):
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        if selected_users:
                            st.write("**👥 Filtered Users:**")
                            for user in selected_users:
                                st.write(f"• {user}")

                    with col2:
                        if selected_branches:
                            st.write("**🌿 Filtered Git Branches:**")
                            for branch in selected_branches:
                                st.write(f"• {branch}")

                    with col3:
                        st.metric(
                            "📊 Filtered Results",
                            f"{len(experiments)} selected",
                            f"from {filter_metadata['filtered_experiments']} matching filters "
                            f"({filter_metadata['total_experiments']} total)",
                        )

            # Create main tabs
            tab1, tab2 = st.tabs(["🔬 Experiment Comparison", "📋 Detailed Analysis"])

            with tab1:
                self.render_main_comparison(experiments)

            with tab2:
                self.render_experiment_details(experiments)
        else:
            st.info("Please select experiments from the sidebar to begin analysis.")

            # Show example of directory structure
            st.subheader("📁 Expected Repo Structure (After Exp Run)")
            st.code(
                """
            data/
            ├── <dataset1>/
            │   ├── input/
            │   │   ├── doc1/
            │   │   └── ...
            │   └── gt/
            │       ├── doc1/
            │       └── ...
            └── ...
            outputs/
            ├── metrics.json
            ├── operational_metrics.json
            ├── postprocess_result.json
            └── timing_results.json
            dvc.yaml
            params.yaml
            """,
                language="text",
            )


# Main application entry point
if __name__ == "__main__":
    dashboard = DashboardUI()
    dashboard.run()
