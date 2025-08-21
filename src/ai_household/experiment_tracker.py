"""Experiment tracking utility for logging and retrieving scientific experiment results.

This module provides simple, file-based experiment tracking with structured metadata
and raw data storage.
"""

from __future__ import annotations

import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


def _convert_numpy_types(data: Any) -> Any:
    """
    Recursively traverses a data structure and converts NumPy numeric types
    to native Python types for clean YAML serialization.
    Updated for NumPy 2.0 compatibility.
    """
    if isinstance(data, dict):
        # If it's a dictionary, recurse on its values
        return {key: _convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, recurse on its elements
        return [_convert_numpy_types(item) for item in data]
    # --- FIX for NumPy 2.0 ---
    # Use np.integer and np.floating, which are the abstract base classes
    # for all NumPy integer and float types, respectively.
    # The aliases np.int_ and np.float_ were removed in NumPy 2.0.
    elif isinstance(data, np.integer):
        # If it's any kind of NumPy integer, convert to a native Python int
        return int(data)
    elif isinstance(data, np.floating):
        # If it's any kind of NumPy float, convert to a native Python float
        return float(data)
    elif isinstance(data, np.ndarray):
        # If it's a NumPy array, convert it to a list
        return data.tolist()
    else:
        # Otherwise, return the data as is
        return data


def log_experiment(
    experiment_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    results_df: pd.DataFrame,
    base_dir: str = "results",
    **metadata_fields: Any,
) -> None:
    """Log experiment results to a structured directory.

    Creates a unique timestamped directory containing metadata and raw results
    for the given experiment run. Accepts any additional metadata fields.
    This version automatically converts NumPy types for clean YAML output.

    Args:
        experiment_name: Name of the experiment
        params: Dictionary of input parameters for the run
        metrics: Dictionary of key summary results from the run
        results_df: DataFrame containing raw, row-level experiment data
        base_dir: Base directory to store results (defaults to "results")
        **metadata_fields: Any additional metadata fields to store (e.g., description, exclusions, inclusions)
    """
    # Create timestamp for unique directory naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create directory name following the required pattern
    dir_name = f"{timestamp}_{experiment_name}"

    # Create the full directory path
    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata structure with flexible fields
    metadata = {
        "parameters": params,
        "metrics": metrics,
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
    }

    # Add any additional metadata fields
    metadata.update(metadata_fields)

    # Convert all NumPy types to native Python types for clean serialization
    clean_metadata = _convert_numpy_types(metadata)

    # Save metadata as YAML file using the cleaned data
    metadata_path = experiment_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        # Use a SafeDumper for better security and portability
        yaml.dump(
            clean_metadata,
            f,
            default_flow_style=False,
            indent=2,
            Dumper=yaml.SafeDumper,
        )

    # Save raw results as pickle
    results_path = experiment_dir / "results.pkl"
    results_df.to_pickle(results_path)

    # Print confirmation message
    print(f"Experiment saved to: {experiment_dir.absolute()}")


def load_all_experiments(base_dir: str = "results") -> pd.DataFrame:
    """Load and aggregate metadata from all experiment runs.

    Scans the base directory for all metadata.yaml files and constructs
    a summary DataFrame with one row per experiment run.

    Args:
        base_dir: Directory to scan for experiment results (defaults to "results")

    Returns:
        DataFrame with columns for all parameters and metrics from the experiments,
        plus a 'run_path' column containing the path to each experiment directory

    Example:
        >>> summary_df = load_all_experiments()
        >>> print(summary_df.columns)
        Index(['experiment_name', 'timestamp', 'windfall_amount', 'num_households',
               'avg_mpc', 'std_mpc', 'run_path'], dtype='object')
    """
    base_path = Path(base_dir)

    # Check if base directory exists
    if not base_path.exists():
        print(f"Warning: Base directory '{base_dir}' does not exist.")
        return pd.DataFrame()

    # Find all metadata.yaml files
    metadata_files = list(base_path.glob("*/metadata.yaml"))

    if not metadata_files:
        print(f"No experiment metadata files found in '{base_dir}'.")
        return pd.DataFrame()

    # Collect all experiment data
    all_experiments = []

    for metadata_file in metadata_files:
        try:
            # Load metadata
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f)

            # Flatten the structure
            experiment_data = {}

            # Add top-level metadata
            if "experiment_name" in metadata:
                experiment_data["experiment_name"] = metadata["experiment_name"]
            if "timestamp" in metadata:
                experiment_data["timestamp"] = metadata["timestamp"]
            if "datetime" in metadata:
                experiment_data["datetime"] = metadata["datetime"]

            # Add parameters (flattened)
            if "parameters" in metadata and isinstance(metadata["parameters"], dict):
                for key, value in metadata["parameters"].items():
                    experiment_data[f"param_{key}"] = value

            # Add metrics (flattened)
            if "metrics" in metadata and isinstance(metadata["metrics"], dict):
                for key, value in metadata["metrics"].items():
                    experiment_data[f"metric_{key}"] = value

            # Add any additional metadata fields (excluding the standard ones)
            standard_fields = {
                "parameters",
                "metrics",
                "experiment_name",
                "timestamp",
                "datetime",
            }
            for key, value in metadata.items():
                if key not in standard_fields:
                    experiment_data[f"meta_{key}"] = value

            # Add run path
            experiment_data["run_path"] = str(metadata_file.parent.absolute())

            all_experiments.append(experiment_data)

        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_file}: {e}")
            continue

    if not all_experiments:
        print("No valid experiment metadata could be loaded.")
        return pd.DataFrame()

    # Create DataFrame
    summary_df = pd.DataFrame(all_experiments)

    # Sort by timestamp for chronological order
    if "timestamp" in summary_df.columns:
        summary_df = summary_df.sort_values("timestamp")

    # Reset index
    summary_df = summary_df.reset_index(drop=True)

    print(f"Loaded {len(summary_df)} experiment runs from '{base_dir}'")

    return summary_df


def load_experiment_results(run_path: str) -> Optional[pd.DataFrame]:
    """Load the raw results DataFrame for a specific experiment run.

    Args:
        run_path: Path to the experiment directory (from load_all_experiments)

    Returns:
        DataFrame containing the raw experiment results, or None if not found
    """
    results_path = Path(run_path) / "results.pkl"

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None

    try:
        return pd.read_pickle(results_path)
    except Exception as e:
        print(f"Error loading results from {results_path}: {e}")
        return None


def get_metadata_field(experiments_df: pd.DataFrame, field_name: str) -> pd.Series:
    """Extract a specific metadata field from loaded experiments.

    This function helps retrieve custom metadata fields that were added
    using the flexible **metadata_fields parameter in log_experiment.

    Args:
        experiments_df: DataFrame from load_all_experiments()
        field_name: Name of the metadata field to extract

    Returns:
        Series containing the values for the specified metadata field

    Example:
        >>> experiments = load_all_experiments()
        >>> descriptions = get_metadata_field(experiments, "description")
        >>> exclusions = get_metadata_field(experiments, "exclusions")
    """
    meta_column = f"meta_{field_name}"

    if meta_column not in experiments_df.columns:
        print(f"Warning: Metadata field '{field_name}' not found in experiments.")
        print(
            f"Available metadata fields: {[col.replace('meta_', '') for col in experiments_df.columns if col.startswith('meta_')]}"
        )
        return pd.Series(dtype="object")

    return experiments_df[meta_column]


__all__ = [
    "log_experiment",
    "load_all_experiments",
    "load_experiment_results",
    "get_metadata_field",
]
