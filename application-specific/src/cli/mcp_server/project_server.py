import json
import os
from pathlib import Path
from typing import Any

from .main import arch_mcp


@arch_mcp.tool()
def create_project(
    name: str,
    output_dir: Path = Path.cwd(),
    description: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """Create a new neural network project.

    This function creates a complete project structure with directories for
    configurations, architectures, outputs, and logs, along with default
    project configuration and architecture files.

    Args:
        name: Name of the project to create.
        output_dir: Directory where the project will be created.
        description: Optional description for the project.
        author: Optional author name for the project.

    Returns:
        dict: A dictionary containing status information and project details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with project information including:
                - project_dir: Path where project was created
                - config_file: Path to project.json file
                - arch_file: Path to default architecture file
                - project_config: Complete project configuration

    """
    project_dir = output_dir / name

    if project_dir.exists():
        return {
            "status": "error",
            "message": f"Project '{name}' already exists in {output_dir}",
            "details": {"project_name": name, "output_dir": str(output_dir)},
        }

    # Create project structure
    try:
        project_dir.mkdir(parents=True)
        (project_dir / "configs").mkdir()
        (project_dir / "architectures").mkdir()
        (project_dir / "outputs").mkdir()
        (project_dir / "logs").mkdir()

        # Create project configuration
        project_config = {
            "name": name,
            "description": description or f"Neural network project: {name}",
            "author": author or "NN Generator CLI",
            "created_at": str(Path.cwd()),
            "version": "1.0.0",
            "architecture": {},
            "config": {},
            "status": "created",
        }

        config_file = project_dir / "project.json"
        with open(config_file, "w") as f:
            json.dump(project_config, f, indent=2)

        # Create default architecture file
        default_arch = {
            "layers": [],
            "misc_params": {
                "width": 224,
                "height": 224,
                "channels": 3,
                "num_epochs": 10,
                "batch_size": 32,
                "device": "cpu",
            },
            "optimizer": {},
            "loss_func": {},
            "scheduler": {},
            "pretrained": {"value": None, "index": -1},
        }

        arch_file = project_dir / "configs" / "default_architecture.json"
        with open(arch_file, "w") as f:
            json.dump(default_arch, f, indent=2)

        return {
            "status": "success",
            "message": "Project created successfully",
            "details": {
                "project_dir": str(project_dir),
                "config_file": str(config_file),
                "arch_file": str(arch_file),
                "project_config": project_config,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating project: {e}",
            "details": {"error": str(e)},
        }


@arch_mcp.tool()
def list_projects(
    directory: Path = Path.cwd(),
) -> dict[str, Any]:
    """List all neural network projects in a directory.

    This function searches for project.json files in the specified directory
    and its subdirectories, returning information about each discovered project.

    Args:
        directory: Directory to search for projects.


    Returns:
        dict: A dictionary containing project listing information.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary containing:
                - projects_count: Number of projects found
                - projects: List of project information dictionaries
                - search_directory: Directory that was searched

    """
    projects = []

    if directory.is_file():
        return {
            "status": "error",
            "message": f"{directory} is not a directory",
            "details": {"provided_path": str(directory)},
        }

    # Find all project.json files
    for root, dirs, files in os.walk(directory):
        if "project.json" in files:
            project_file = Path(root) / "project.json"
            try:
                with open(project_file) as f:
                    project_data = json.load(f)

                project_info = {
                    "name": project_data.get("name", "Unknown"),
                    "path": str(Path(root).relative_to(directory)),
                    "description": project_data.get("description", "No description"),
                    "author": project_data.get("author", "Unknown"),
                    "status": project_data.get("status", "unknown"),
                    "created": project_data.get("created_at", "Unknown"),
                }
                projects.append(project_info)

            except Exception:
                # Log error but continue with other projects
                pass

    if not projects:
        return {
            "status": "success",
            "message": f"No projects found in {directory}",
            "details": {
                "projects_count": 0,
                "projects": [],
                "search_directory": str(directory),
            },
        }

    return {
        "status": "success",
        "message": f"Found {len(projects)} project(s)",
        "details": {
            "projects_count": len(projects),
            "projects": projects,
            "search_directory": str(directory),
        },
    }


@arch_mcp.tool()
def load_project(
    project_path: Path,
    set_active: bool = True,
) -> dict[str, Any]:
    """Load an existing neural network project.

    This function loads project configuration from a project.json file
    and validates the project structure. Optionally sets the project as active.

    Args:
        project_path: Path to project directory or project.json file.
        set_active: Whether to set this project as the active project.

    Returns:
        dict: A dictionary containing status information and project details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with project information including:
                - project_name: Name of the loaded project
                - project_dir: Directory containing the project
                - project_file: Path to project.json file
                - project_data: Complete project configuration
                - set_as_active: Whether project was set as active

    """
    # Handle both directory and file inputs
    if project_path.is_file():
        if project_path.name == "project.json":
            project_dir = project_path.parent
            project_file = project_path
        else:
            return {
                "status": "error",
                "message": f"{project_path} is not a project.json file",
                "details": {"provided_path": str(project_path)},
            }
    elif project_path.is_dir():
        project_file = project_path / "project.json"
        if not project_file.exists():
            return {
                "status": "error",
                "message": f"No project.json found in {project_path}",
                "details": {"project_path": str(project_path)},
            }
        project_dir = project_path
    else:
        return {
            "status": "error",
            "message": f"Project path {project_path} does not exist",
            "details": {"project_path": str(project_path)},
        }

    try:
        with open(project_file) as f:
            project_data = json.load(f)

        # Validate project structure
        required_keys = ["name", "architecture", "config"]
        missing_keys = [key for key in required_keys if key not in project_data]

        if missing_keys:
            return {
                "status": "error",
                "message": f"Invalid project file. Missing keys: {missing_keys}",
                "details": {
                    "missing_keys": missing_keys,
                    "project_file": str(project_file),
                },
            }

        # Load project data
        project_name = project_data["name"]
        description = project_data.get("description", "No description")
        author = project_data.get("author", "Unknown")

        return {
            "status": "success",
            "message": "Project loaded successfully",
            "details": {
                "project_name": project_name,
                "project_dir": str(project_dir),
                "project_file": str(project_file),
                "project_data": project_data,
                "set_as_active": set_active,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading project: {e}",
            "details": {"error": str(e)},
        }


@arch_mcp.tool()
def project_info(
    project_path: Path,
) -> dict[str, Any]:
    """Show detailed information about a project.

    This function provides comprehensive information about a neural network project
    including statistics about configurations, architectures, and outputs.

    Args:
        project_path: Path to project directory or project.json file.

    Returns:
        dict: A dictionary containing detailed project information.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with project information including:
                - project_name: Name of the project
                - project_dir: Directory containing the project
                - project_file: Path to project.json file
                - project_data: Complete project configuration
                - statistics: Dictionary with counts of configs, architectures, and outputs

    """
    # Handle both directory and file inputs
    if project_path.is_file():
        if project_path.name == "project.json":
            project_file = project_path
        else:
            return {
                "status": "error",
                "message": f"{project_path} is not a project.json file",
                "details": {"provided_path": str(project_path)},
            }
    elif project_path.is_dir():
        project_file = project_path / "project.json"
        if not project_file.exists():
            return {
                "status": "error",
                "message": f"No project.json found in {project_path}",
                "details": {"project_path": str(project_path)},
            }
    else:
        return {
            "status": "error",
            "message": f"Project path {project_path} does not exist",
            "details": {"project_path": str(project_path)},
        }

    try:
        with open(project_file) as f:
            project_data = json.load(f)

        # Get project statistics
        project_dir = project_file.parent
        num_configs = (
            len(list((project_dir / "configs").glob("*.json")))
            if (project_dir / "configs").exists()
            else 0
        )
        num_architectures = (
            len(list((project_dir / "architectures").glob("*.json")))
            if (project_dir / "architectures").exists()
            else 0
        )
        num_outputs = (
            len(list((project_dir / "outputs").iterdir()))
            if (project_dir / "outputs").exists()
            else 0
        )

        statistics = {
            "configs_count": num_configs,
            "architectures_count": num_architectures,
            "outputs_count": num_outputs,
        }

        return {
            "status": "success",
            "message": "Project information retrieved successfully",
            "details": {
                "project_name": project_data.get("name", "Unknown"),
                "project_dir": str(project_dir),
                "project_file": str(project_file),
                "project_data": project_data,
                "statistics": statistics,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading project info: {e}",
            "details": {"error": str(e)},
        }
