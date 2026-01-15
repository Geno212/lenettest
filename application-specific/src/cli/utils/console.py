"""utils.console.

Centralized Rich console utilities for CLI output.

This module provides:
- A shared Rich Console instance for consistent styling
- Verbosity toggles to control output levels
- Dry-run support to simulate actions without executing them
- Helper functions for styled output (success, error, info, warning)

Usage:
    from src.cli.utils.console import console, print_success, set_verbosity, set_dry_run
"""

from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

# ─── Console Setup ─────────────────────────────────────────────────────────────

custom_theme = Theme(
    {
        "success": "bold green",
        "error": "bold red",
        "info": "bold cyan",
        "warning": "bold yellow",
        "dryrun": "dim italic",
    },
)

console: Console = Console(theme=custom_theme)
error_console: Console = Console(stderr=True, theme=custom_theme)

# ─── Verbosity and Dry-Run Flags ──────────────────────────────────────────────

_verbose: bool = False
_dry_run: bool = False


def set_verbosity(enabled: bool) -> None:
    """Enable or disable verbose output.

    Args:
        enabled (bool): True to enable verbose mode, False to disable.

    """
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    """Check if verbose mode is enabled.

    Returns:
        bool: True if verbose mode is active, False otherwise.

    """
    return _verbose


def set_dry_run(enabled: bool) -> None:
    """Enable or disable dry-run mode.

    Args:
        enabled (bool): True to enable dry-run mode, False to disable.

    """
    global _dry_run
    _dry_run = enabled


def is_dry_run() -> bool:
    """Check if dry-run mode is enabled.

    Returns:
        bool: True if dry-run mode is active, False otherwise.

    """
    return _dry_run


# ─── Output Helpers ───────────────────────────────────────────────────────────


def print_success(msg: str) -> None:
    """Print a success message to stdout with green styling.

    Args:
        msg (str): The message to display.

    """
    console.print(f"[success]✔ {escape(msg)}[/success]")


def print_error(msg: str) -> None:
    """Print an error message to stderr with red styling.

    Args:
        msg (str): The error message to display.

    """
    error_console.print(f"[error]✖ {escape(msg)}[/error]")


def print_critical(msg: str) -> None:
    """Print a critical error message to stderr with bold red underline.

    Args:
        msg (str): The critical message to display.

    """
    error_console.print(f"[error][bold underline]CRITICAL:[/] {escape(msg)}[/error]")


def print_info(msg: str) -> None:
    """Print an informational message to stdout with cyan styling.

    Args:
        msg (str): The info message to display.

    """
    console.print(f"[info]ℹ {escape(msg)}[/info]")


def print_warning(msg: str) -> None:
    """Print a warning message to stdout with yellow styling.

    Args:
        msg (str): The warning message to display.

    """
    console.print(f"[warning]! {escape(msg)}[/warning]")


def print_dryrun(msg: str) -> None:
    """Print a dry-run message if dry-run mode is enabled.

    Args:
        msg (str): The message describing the simulated action.

    """
    if _dry_run:
        console.print(f"[dryrun]DRY-RUN: {escape(msg)}[/dryrun]")


def print_verbose(msg: str) -> None:
    """Print a verbose message if verbosity is enabled.

    Args:
        msg (str): The detailed message to display.

    """
    if _verbose:
        console.print(f"[dim]{escape(msg)}[/dim]")
