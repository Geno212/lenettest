"""
System Configuration Module

This module handles all system configuration including:
- LLM provider settings
- MCP server connection
- Default values
- User preferences
- Workflow settings

Configuration is loaded from:
1. Environment variables (highest priority)
2. Config file (~/.nn_generator/config.yaml)
3. Default values (lowest priority)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    
    provider: str = "gemini"
    """LLM provider: 'openai', 'anthropic', or 'local'"""
    
    model: str = "gemini-2.0-flash-exp"
    """Model name to use for assistants"""
    
    temperature: float = 0.0
    """Temperature for LLM responses (0 = deterministic, 1 = creative)"""
    
    max_tokens: int = 4000
    """Maximum tokens per response"""
    
    api_key: Optional[str] = None
    """API key for LLM provider (loaded from env if not set)"""


@dataclass
class MCPConfig:
    """MCP server configuration."""
    
    server_url: str = "http://127.0.0.1:8000/mcp"
    """URL of the MCP server"""
    
    timeout: int = 300
    """Request timeout in seconds"""
    
    retry_attempts: int = 3
    """Number of retry attempts for failed requests"""
    
    retry_delay: float = 1.0
    """Delay between retry attempts in seconds"""


@dataclass
class LangSmithConfig:
    """LangSmith tracing configuration."""
    
    enabled: bool = False
    """Enable LangSmith tracing (requires API key)"""
    
    api_key: Optional[str] = None
    """LangSmith API key (loaded from env LANGSMITH_API_KEY)"""
    
    project_name: str = "nn-generator-agentic"
    """LangSmith project name for organizing traces"""
    
    endpoint: str = "https://api.smith.langchain.com"
    """LangSmith API endpoint"""
    
    tracing_sample_rate: float = 1.0
    """Sampling rate for traces (0.0 to 1.0, 1.0 = trace everything)"""


@dataclass
class DefaultValues:
    """Default values for various parameters."""
    

    
    device: str = "cpu"
    """Default device: 'cpu', 'cuda', 'mps'"""
    
    epochs: int = 50
    """Default number of training epochs"""
    
    batch_size: int = 32
    """Default batch size"""
    
    learning_rate: float = 0.001
    """Default learning rate"""
    
    optimizer: str = "Adam"
    """Default optimizer"""
    
    max_optimization_iterations: int = 3
    """Maximum optimization iterations"""
    
    min_improvement_threshold: float = 0.01
    """Minimum improvement to continue optimization (1%)"""


@dataclass
class UserPreferences:
    """User preferences for workflow behavior."""
    
    confirm_before_training: bool = True
    """Ask for confirmation before starting training"""
    
    auto_start_tensorboard: bool = True
    """Automatically start TensorBoard when training begins"""
    
    save_checkpoints: bool = True
    """Save model checkpoints during training"""
    
    checkpoint_frequency: int = 5
    """Save checkpoint every N epochs"""
    
    enable_interrupts: bool = True
    """Enable human-in-the-loop interrupts for sensitive operations"""
    
    verbose_logging: bool = False
    """Enable detailed logging"""


@dataclass
class WorkflowSettings:
    """Workflow-specific settings."""
    
    enable_streaming: bool = True
    """Stream LLM responses for better UX"""
    
    max_conversation_history: int = 100
    """Maximum messages to keep in conversation history"""
    
    auto_save_interval: int = 5
    """Auto-save state every N steps"""
    
    enable_state_persistence: bool = True
    """Enable saving and resuming workflows"""


@dataclass
class SystemConfig:
    """
    Complete system configuration.
    
    This class manages all configuration for the NN Generator system.
    Configuration is loaded from multiple sources with priority:
    1. Environment variables (highest)
    2. Config file (~/.nn_generator/config.yaml)
    3. Defaults (lowest)
    
    Usage:
        # Load from default locations
        config = SystemConfig.load()
        
        # Load from specific file
        config = SystemConfig.load(config_file="path/to/config.yaml")
        
        # Access configuration
        print(config.llm.model)
        print(config.defaults.epochs)
    """
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)
    defaults: DefaultValues = field(default_factory=DefaultValues)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    workflow: WorkflowSettings = field(default_factory=WorkflowSettings)
    
    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "SystemConfig":
        """
        Load configuration from file and environment variables.
        
        Priority (highest to lowest):
        1. Environment variables
        2. Config file
        3. Default values
        
        Args:
            config_file: Optional path to config file.
                        If None, uses ~/.nn_generator/config.yaml
        
        Returns:
            SystemConfig instance with merged configuration
        
        Example:
            config = SystemConfig.load()
            config = SystemConfig.load(config_file="./custom_config.yaml")
        """
        # Start with defaults
        config = cls()
        
        # Override with environment variables
        config._load_from_env()
        
        return config
      
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        
        # LLM configuration
        # Support both GOOGLE_API_KEY (Gemini) and OPENAI_API_KEY (backwards compatibility)
        if "GOOGLE_API_KEY" in os.environ:
            self.llm.api_key = os.environ["GOOGLE_API_KEY"]
        # elif "OPENAI_API_KEY" in os.environ:
        #     self.llm.api_key = os.environ["OPENAI_API_KEY"]
        
        if "LLM_PROVIDER" in os.environ:
            self.llm.provider = os.environ["LLM_PROVIDER"]
        
        if "LLM_MODEL" in os.environ:
            self.llm.model = os.environ["LLM_MODEL"]
        
        if "LLM_TEMPERATURE" in os.environ:
            self.llm.temperature = float(os.environ["LLM_TEMPERATURE"])
        
        # MCP configuration
        if "MCP_SERVER_URL" in os.environ:
            self.mcp.server_url = os.environ["MCP_SERVER_URL"]
        
        if "MCP_TIMEOUT" in os.environ:
            self.mcp.timeout = int(os.environ["MCP_TIMEOUT"])
        
        # LangSmith configuration
        if "LANGSMITH_API_KEY" in os.environ:
            self.langsmith.api_key = os.environ["LANGSMITH_API_KEY"]
            # Auto-enable tracing if API key is present
            self.langsmith.enabled = True
        
        if "LANGSMITH_PROJECT" in os.environ:
            self.langsmith.project_name = os.environ["LANGSMITH_PROJECT"]
        
        if "LANGSMITH_ENDPOINT" in os.environ:
            self.langsmith.endpoint = os.environ["LANGSMITH_ENDPOINT"]
        
        if "LANGSMITH_TRACING" in os.environ:
            self.langsmith.enabled = os.environ["LANGSMITH_TRACING"].lower() in ["true", "1", "yes"]
        
        # Device override
        if "DEVICE" in os.environ:
            self.defaults.device = os.environ["DEVICE"]
    
   
    



# Convenience function to load configuration
def load_config(config_file: Optional[str] = None) -> SystemConfig:
    """
    Load system configuration.
    
    Convenience function that wraps SystemConfig.load().
    
    Args:
        config_file: Optional path to config file
    
    Returns:
        SystemConfig instance
    
    Example:
        from src.core.config import load_config
        config = load_config()
    """
    return SystemConfig.load(config_file)