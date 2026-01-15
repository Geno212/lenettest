#!/usr/bin/env python3
"""Interactive Shell for Neural Network Generator CLI

This module provides an interactive shell that allows running multiple CLI commands
in a single Python process, preserving state between commands.
"""

# Add the src directory to Python path for proper imports
import shlex
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .commands.architecture import app as architecture_app
from .commands.config import app as config_app
from .commands.generate import app as generate_app
from .commands.project import app as project_app
from .commands.tensorboard import app as tensorboard_app
from .commands.test import app as test_app
from .commands.train import app as train_app

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import command groups
from .commands.validate import app as validate_app

console = Console()


class NNGeneratorShell:
    """Interactive shell for the Neural Network Generator CLI."""

    def __init__(self):
        self.running = True

    def print_welcome(self):
        """Print welcome message and available commands."""
        welcome_text = Text.assemble(
            ("Neural Network Generator Shell", "bold blue"),
            ("\nType commands just like you would in the CLI"),
            ("\nAvailable command groups: ", "dim"),
            ("project, arch, config, train, gen, validate, test, tensorboard", "cyan"),
            ("\nSpecial commands: ", "dim"),
            ("help, exit, quit", "cyan"),
            ("\n\nExample: ", "dim"),
            (
                "config model-params --height 224 --width 224 --epochs 10 --batch-size 32 --dataset MNIST --dataset-path /path/to/mnist",
                "green",
            ),
        )

        console.print(
            Panel(
                welcome_text,
                title="[bold]NN Generator Shell[/bold]",
                border_style="blue",
            ),
        )

    def print_help(self):
        """Print help information."""
        help_text = Text.assemble(
            ("Available Commands:", "bold blue"),
            ("\n\nproject", "cyan"),
            (" - Project management commands", "dim"),
            ("\narch", "cyan"),
            (" - Architecture design commands", "dim"),
            ("\nconfig", "cyan"),
            (" - Configuration and parameters commands", "dim"),
            ("\ntrain", "cyan"),
            (" - Model training commands", "dim"),
            ("\ngen", "cyan"),
            (" - Code generation commands", "dim"),
            ("\nvalidate", "cyan"),
            (" - Architecture validation commands", "dim"),
            ("\ntest", "cyan"),
            (" - Model testing commands", "dim"),
            ("\ntensorboard", "cyan"),
            (" - TensorBoard management commands", "dim"),
            ("\n\nhelp", "cyan"),
            (" - Show this help message", "dim"),
            ("\nexit/quit", "cyan"),
            (" - Exit the shell", "dim"),
        )
        console.print(
            Panel(help_text, title="[bold]Shell Commands[/bold]", border_style="green"),
        )

    def dispatch_command(self, args: list[str]):
        """Dispatch command to appropriate handler."""
        if not args:
            return

        command = args[0]

        try:
            if command in {"exit", "quit"}:
                self.running = False
                console.print("[bold green]👋 Goodbye![/bold green]")
                return
            if command == "help":
                self.print_help()
                return

            # Dispatch to Typer apps
            if command == "project":
                project_app(args[1:])
            elif command == "arch":
                architecture_app(args[1:])
            elif command == "config":
                config_app(args[1:])
            elif command == "train":
                train_app(args[1:])
            elif command == "gen":
                generate_app(args[1:])
            elif command == "validate":
                validate_app(args[1:])
            elif command == "test":
                test_app(args[1:])
            elif command == "tensorboard":
                tensorboard_app(args[1:])
            else:
                console.print(f"[red]❌ Unknown command: {command}[/red]")
                console.print("[yellow]Use 'help' to see available commands[/yellow]")
        except SystemExit:
            # Typer commands may call sys.exit(), which we catch to keep the shell running
            pass
        except Exception as e:
            console.print(f"[red]❌ Error executing command: {e}[/red]")

    def run(self):
        """Run the interactive shell."""
        self.print_welcome()

        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask(
                    "\n[bold blue]nn-generator>[/bold blue]",
                    default="",
                )

                # Handle empty input
                if not user_input.strip():
                    continue

                # Parse command
                args = shlex.split(user_input)

                # Dispatch command
                self.dispatch_command(args)

            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]⚠️  Use 'exit' or 'quit' to exit the shell[/bold yellow]",
                )
            except EOFError:
                console.print("\n[bold green]👋 Goodbye![/bold green]")
                break


def main():
    """Main entry point for the shell."""
    shell = NNGeneratorShell()
    shell.run()


if __name__ == "__main__":
    main()
