"""Utility functions for loading and validating configuration files for the CLI."""

import json
from pathlib import Path

import typer


def validate_json_path_param(file: Path | None) -> Path | None:
    """Validate that the given file is a valid JSON file.

    Parameters
    ----------
    file : Path
        Path to the JSON file to validate.

    Returns
    -------
    Path
        The original file path if the file is valid JSON.

    Raises
    ------
    typer.BadParameter
        If the file is not valid JSON or cannot be read.

    """
    if file is None:
        return file

    try:
        with file.open("r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON: {e}"
        raise typer.BadParameter(msg) from e
    except Exception as e:
        msg = f"Error reading file: {e}"
        raise typer.BadParameter(msg) from e
    return file
