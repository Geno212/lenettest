# src/logic/project_logic.py (continued)
"""Deterministic logic for project management."""

import os
import re
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ProjectManagerLogic:
    """
    Deterministic logic for project management operations.
    
    Handles:
    - Name sanitization
    - Path validation
    - Project structure definition
    - Name generation
    """
    
    def sanitize_project_name(self, name: str) -> str:
        """
        Sanitize project name for filesystem compatibility.
        
        Rules:
        - Replace spaces with underscores
        - Remove special characters
        - Lowercase
        - Remove multiple underscores
        - Max 50 characters
        
        Args:
            name: Raw project name
            
        Returns:
            Sanitized name safe for filesystem
            
        Example:
            >>> sanitize_project_name("My Cool Project!")
            "my_cool_project"
        """
        # Replace spaces with underscores
        sanitized = name.replace(" ", "_")
        
        # Remove special characters (keep alphanumeric and underscore)
        sanitized = re.sub(r'[^\w\-]', '_', sanitized)
        
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Lowercase
        sanitized = sanitized.lower()
        
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50].rstrip('_')
        
        # Ensure not empty
        if not sanitized:
            sanitized = "nn_project"
        
        return sanitized
    
    def generate_project_name(self, task_spec: Dict[str, Any]) -> str:
        """
        Generate project name from task specification.
        
        Format: {dataset}_{task_type}_project
        
        Args:
            task_spec: Task specification dictionary
            
        Returns:
            Generated project name
            
        Example:
            >>> generate_project_name({"dataset": {"name": "MNIST"}, "task_type": "classification"})
            "mnist_classification_project"
        """
        dataset_name = task_spec.get("dataset", {}).get("name", "custom")
        task_type = task_spec.get("task_type", "model")
        
        name = f"{dataset_name}_{task_type}_project"
        return self.sanitize_project_name(name)
    
    def validate_project_path(self, output_dir: str, project_name: str) -> Dict[str, Any]:
        """
        Validate project path for creation.
        
        Checks:
        - Directory exists
        - Is a directory (not a file)
        - Writable permissions
        - Absolute path
        
        Args:
            output_dir: Parent directory
            project_name: Project name
            
        Returns:
            Dictionary with validation results:
            {
                "exists": bool,
                "is_directory": bool,
                "writable": bool,
                "absolute_path": str
            }
        """
        # Expand user home directory
        expanded_dir = os.path.expanduser(output_dir)
        
        # Create full path
        full_path = Path(expanded_dir) / project_name
        
        return {
            "exists": full_path.exists(),
            "is_directory": full_path.is_dir() if full_path.exists() else None,
            "writable": os.access(Path(expanded_dir), os.W_OK) if Path(expanded_dir).exists() else False,
            "absolute_path": str(full_path.absolute()),
            "parent_exists": Path(expanded_dir).exists()
        }
    
    def get_project_path(self, output_dir: str, project_name: str) -> str:
        """
        Get absolute project path.
        
        Args:
            output_dir: Parent directory
            project_name: Project name
            
        Returns:
            Absolute path string
        """
        expanded_dir = os.path.expanduser(output_dir)
        full_path = Path(expanded_dir) / project_name
        return str(full_path.absolute())
    
    def get_project_structure(self, project_name: str = "project_name") -> Dict[str, Any]:
        """
        Get expected project directory structure.
        
        Returns:
            Dictionary describing structure
        """
        return {
            "name": project_name,
            "subdirs": [
                "architectures",
                "configs",
                "generated_code",
                "training_outputs",
                "logs",
                "models",
                "reports"
            ],
            "description": "Standard neural network project structure"
        }
    
    def get_timestamp(self) -> str:
        """
        Get timestamp string for unique naming.
        
        Format: YYYYMMDD_HHMMSS
        
        Returns:
            Timestamp string
            
        Example:
            "20240115_143022"
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def suggest_alternative_name(self, base_name: str) -> str:
        """
        Suggest alternative project name with timestamp.
        
        Args:
            base_name: Original project name
            
        Returns:
            Alternative name with timestamp
            
        Example:
            >>> suggest_alternative_name("my_project")
            "my_project_20240115_143022"
        """
        timestamp = self.get_timestamp()
        return f"{base_name}_{timestamp}"