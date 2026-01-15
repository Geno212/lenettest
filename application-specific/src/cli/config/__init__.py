#!/usr/bin/env python3
"""Configuration System

Manages CLI configuration and settings.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

console = Console()


class ConfigManager:
    """Manages CLI configuration."""

    def __init__(self, config_file: Path | None = None):
        if config_file is None:
            # Default config location
            config_dir = Path(__file__).parent
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "config.json"

        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return self.get_default_config()

        try:
            with open(self.config_file) as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]⚠️  Error loading config: {e}[/yellow]")
            return self.get_default_config()

    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[red]❌ Error saving config: {e}[/red]")

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "default_device": "cpu",
            "default_batch_size": 32,
            "default_epochs": 10,
            "output_directory": str(Path.cwd() / "outputs"),
            "log_level": "info",
            "auto_save": True,
            "theme": "default",
            "parallel_processing": False,
            "max_workers": 4,
            "recent_projects": [],
            "favorite_architectures": [],
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
        self.save_config()

    def update(self, updates: dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)
        self.save_config()


# Global configuration manager
config_manager = ConfigManager()
