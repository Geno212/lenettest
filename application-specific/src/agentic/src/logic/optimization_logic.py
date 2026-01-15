# src/logic/optimization_logic.py
"""Deterministic logic for optimization analysis and strategy generation."""

from typing import Dict, Any, List, Optional
import statistics


class OptimizationAdvisorLogic:
    """
    Deterministic logic for optimization operations.
    
    Handles:
    - Training curve analysis
    - Issue identification  
    - Strategy generation
    - Performance tracking
    """
    
    # Issue categories
    UNDERFITTING = "underfitting"
    OVERFITTING = "overfitting"
    NON_CONVERGENCE = "non_convergence"
    SLOW_CONVERGENCE = "slow_convergence"
    GRADIENT_ISSUES = "gradient_issues"
    
    # Severity levels
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    def analyze_training_curves(self, training_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze training curves to understand model behavior.
        
        Args:
            training_runs: List of training run dictionaries
            
        Returns:
            Analysis with status, trends, convergence info
        """
        if not training_runs:
            return {
                "status": "no_data",
                "converged": False,
                "trends": {},
                "issues": []
            }
        
        latest_run = training_runs[-1]
        
        # Extract metrics
        final_accuracy = latest_run.get("final_accuracy", 0)
        best_accuracy = latest_run.get("best_accuracy", final_accuracy)
        final_loss = latest_run.get("final_loss", float('inf'))
        converged = latest_run.get("converged", False)
        
        # Determine status
        status = self._determine_status(final_accuracy)
        
        # Analyze trends across runs
        trends = self._analyze_trends(training_runs)
        
        return {
            "status": status,
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "final_loss": final_loss,
            "converged": converged,
            "trends": trends,
            "run_count": len(training_runs)
        }
    
    def _determine_status(self, accuracy: float) -> str:
        """Determine overall training status from accuracy."""
        if accuracy == 0:
            return "failed"
        elif accuracy < 0.5:
            return "poor"
        elif accuracy < 0.7:
            return "below_average"
        elif accuracy < 0.85:
            return "good"
        elif accuracy < 0.95:
            return "very_good"
        else:
            return "excellent"
    
    def _analyze_trends(self, training_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trends across multiple training runs.
        
        Returns trends like improving, plateauing, degrading.
        """
        if len(training_runs) < 2:
            return {"trend": "insufficient_data"}
        
        accuracies = [run.get("final_accuracy", 0) for run in training_runs]
        
        # Calculate improvement
        improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
        
        # Determine trend
        if all(imp > 0.01 for imp in improvements):
            trend = "improving"
        elif all(imp < 0.01 for imp in improvements):
            trend = "plateauing"
        elif improvements[-1] < 0:
            trend = "degrading"
        else:
            trend = "mixed"
        
        # Calculate average improvement
        avg_improvement = statistics.mean(improvements) if improvements else 0
        
        return {
            "trend": trend,
            "improvements": improvements,
            "avg_improvement": avg_improvement,
            "total_improvement": accuracies[-1] - accuracies[0]
        }
    
    def identify_issues(
        self,
        analysis: Dict[str, Any],
        target_accuracy: Optional[float],
        current_accuracy: float
    ) -> List[Dict[str, Any]]:
        """
        Identify performance issues from analysis.
        
        Args:
            analysis: Training analysis results
            target_accuracy: Target accuracy goal
            current_accuracy: Current achieved accuracy
            
        Returns:
            List of identified issues with category, description, severity
        """
        issues = []
        
        # Calculate gap if target provided
        gap = (target_accuracy - current_accuracy) if target_accuracy else 0
        
        # Issue detection logic
        
        # 1. Check for underfitting
        if current_accuracy < 0.7:
            issues.append({
                "category": self.UNDERFITTING,
                "description": f"Model accuracy ({current_accuracy*100:.1f}%) is low, indicating underfitting",
                "severity": self.HIGH if current_accuracy < 0.5 else self.MEDIUM,
                "evidence": f"Both training and validation accuracies are below 70%"
            })
        
        # 2. Check for slow progress
        if analysis["trends"].get("trend") == "plateauing" and gap > 0.05:
            issues.append({
                "category": self.SLOW_CONVERGENCE,
                "description": "Model improvement has plateaued before reaching target",
                "severity": self.MEDIUM,
                "evidence": f"Last few runs showed minimal improvement (<1%)"
            })
        
        # 3. Check for degradation
        if analysis["trends"].get("trend") == "degrading":
            issues.append({
                "category": self.OVERFITTING,
                "description": "Model performance is degrading, likely overfitting",
                "severity": self.HIGH,
                "evidence": "Performance decreased in recent iteration"
            })
        
        # 4. Check for non-convergence (based on status)
        if analysis.get("status") == "failed":
            issues.append({
                "category": self.NON_CONVERGENCE,
                "description": "Model failed to train properly",
                "severity": self.CRITICAL,
                "evidence": "Accuracy is 0% or near random chance"
            })
        
        # 5. If no obvious issues but target not met
        if not issues and target_accuracy and current_accuracy < target_accuracy:
            if gap > 0.1:
                issues.append({
                    "category": self.UNDERFITTING,
                    "description": f"Model is {gap*100:.1f}% below target, may need more capacity",
                    "severity": self.MEDIUM,
                    "evidence": f"Gap to target: {gap*100:.1f}%"
                })
            else:
                issues.append({
                    "category": "fine_tuning_needed",
                    "description": f"Model is close to target (within {gap*100:.1f}%), needs fine-tuning",
                    "severity": self.LOW,
                    "evidence": f"Only {gap*100:.1f}% away from target"
                })
        
        return issues
    
    def generate_optimization_strategy(
        self,
        issues: List[Dict[str, Any]],
        current_config: Dict[str, Any],
        current_arch: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate optimization strategy based on identified issues.
        
        Args:
            issues: List of identified issues
            current_config: Current training configuration
            current_arch: Current architecture summary
            analysis: Training analysis
            
        Returns:
            Strategy dictionary with type, changes, reasoning
        """
        if not issues:
            return {
                "type": "none",
                "changes": {},
                "reasoning": "No clear issues identified"
            }
        
        # Get primary issue (highest severity)
        primary_issue = self._get_primary_issue(issues)
        category = primary_issue["category"]
        
        # Generate strategy based on issue category
        if category == self.UNDERFITTING:
            return self._strategy_for_underfitting(current_config, current_arch)
        
        elif category == self.OVERFITTING:
            return self._strategy_for_overfitting(current_config, current_arch)
        
        elif category == self.NON_CONVERGENCE:
            return self._strategy_for_non_convergence(current_config)
        
        elif category == self.SLOW_CONVERGENCE:
            return self._strategy_for_slow_convergence(current_config)
        
        elif category == "fine_tuning_needed":
            return self._strategy_for_fine_tuning(current_config, current_arch)
        
        else:
            return {
                "type": "unknown",
                "changes": {},
                "reasoning": "Issue category not recognized"
            }
    
    def _get_primary_issue(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get the most severe issue."""
        severity_order = {
            self.CRITICAL: 0,
            self.HIGH: 1,
            self.MEDIUM: 2,
            self.LOW: 3
        }
        
        return min(issues, key=lambda x: severity_order.get(x["severity"], 99))
    
    def _strategy_for_underfitting(
        self,
        config: Dict[str, Any],
        arch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategy for underfitting."""
        
        # Decision: Architecture change (add capacity) or config change (train longer/higher LR)
        
        # If small architecture, increase capacity
        total_params = arch.get("parameters", 0)
        
        if total_params < 1_000_000:  # Less than 1M params
            return {
                "type": "architecture",
                "changes": {
                    "action": "increase_capacity",
                    "suggestion": "Add 2 more convolutional layers or increase filters by 50%"
                },
                "reasoning": "Model has low capacity (< 1M parameters). Adding layers will increase learning ability."
            }
        else:
            # Try training longer or higher LR
            current_epochs = config.get("epochs", 50)
            current_lr = config.get("optimizer", {}).get("lr", 0.001)
            
            return {
                "type": "configuration",
                "changes": {
                    "epochs": min(current_epochs * 2, 200),
                    "optimizer": {
                        **config.get("optimizer", {}),
                        "lr": min(current_lr * 2, 0.01)
                    }
                },
                "reasoning": f"Model has sufficient capacity but may need more training time or higher learning rate to converge."
            }
    
    def _strategy_for_overfitting(
        self,
        config: Dict[str, Any],
        arch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategy for overfitting."""
        
        current_wd = config.get("optimizer", {}).get("weight_decay", 0)
        
        if current_wd < 0.0001:
            # Add regularization via config
            return {
                "type": "configuration",
                "changes": {
                    "optimizer": {
                        **config.get("optimizer", {}),
                        "weight_decay": 0.001
                    }
                },
                "reasoning": "Adding weight decay to penalize model complexity and reduce overfitting."
            }
        else:
            # Add dropout via architecture
            return {
                "type": "architecture",
                "changes": {
                    "action": "add_regularization",
                    "suggestion": "Add Dropout(0.5) after dense layers"
                },
                "reasoning": "Weight decay alone insufficient. Dropout will force model to learn robust features."
            }
    
    def _strategy_for_non_convergence(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy for non-convergence."""
        
        current_lr = config.get("optimizer", {}).get("lr", 0.001)
        current_optimizer = config.get("optimizer", {}).get("type", "Adam")
        
        return {
            "type": "configuration",
            "changes": {
                "optimizer": {
                    "type": "Adam" if current_optimizer == "SGD" else "SGD",
                    "lr": current_lr / 10,
                    "momentum": 0.9 if current_optimizer == "SGD" else None
                }
            },
            "reasoning": f"Reducing learning rate by 10x and switching optimizer to improve stability."
        }
    
    def _strategy_for_slow_convergence(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy for slow convergence."""
        
        current_lr = config.get("optimizer", {}).get("lr", 0.001)
        has_scheduler = config.get("scheduler")
        batch_size = config.get("batch_size", 32)
        
        # If no scheduler, add one
        if not has_scheduler:
            return {
                "type": "configuration",
                "changes": {
                    "scheduler": {
                        "type": "ReduceLROnPlateau",
                        "mode": "min",
                        "factor": 0.5,
                        "patience": 5,
                        "min_lr": 1e-6
                    }
                },
                "reasoning": "Adding learning rate scheduler to adaptively reduce learning rate when validation loss plateaus."
            }
        
        # If batch size is small, try increasing it
        if batch_size < 128:
            return {
                "type": "configuration",
                "changes": {
                    "batch_size": min(batch_size * 2, 256)  # Double batch size, cap at 256
                },
                "reasoning": f"Increasing batch size from {batch_size} to {min(batch_size * 2, 256)} to speed up convergence through more stable gradient estimates."
            }
        
        # If learning rate is too low, increase it
        if current_lr < 1e-4:
            return {
                "type": "configuration",
                "changes": {
                    "optimizer": {
                        **config.get("optimizer", {}),
                        "lr": current_lr * 5  # Increase learning rate by 5x
                    }
                },
                "reasoning": f"Learning rate ({current_lr:.2e}) is very low. Increasing learning rate to {current_lr * 5:.2e} to speed up convergence."
            }
        
        # If we've tried everything, switch to a different optimizer
        current_optimizer = config.get("optimizer", {}).get("type", "Adam")
        new_optimizer = "AdamW" if current_optimizer == "Adam" else "Adam"
        
        return {
            "type": "configuration",
            "changes": {
                "optimizer": {
                    "type": new_optimizer,
                    "lr": 0.001,
                    "weight_decay": 0.01 if new_optimizer == "AdamW" else 0.0
                }
            },
            "reasoning": f"Switching from {current_optimizer} to {new_optimizer} optimizer to potentially improve convergence characteristics."
        }
    
    def _strategy_for_fine_tuning(self, config: Dict[str, Any], arch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate strategy for fine-tuning when close to target accuracy.
        
        This is called when the model is within 10% of the target accuracy.
        Focuses on subtle adjustments to training parameters rather than
        major architectural changes.
        """
        current_lr = config.get("optimizer", {}).get("lr", 0.001)
        batch_size = config.get("batch_size", 32)
        has_scheduler = bool(config.get("scheduler"))
        
        # If no scheduler, add a conservative one
        if not has_scheduler:
            return {
                "type": "configuration",
                "changes": {
                    "scheduler": {
                        "type": "ReduceLROnPlateau",
                        "mode": "min",
                        "factor": 0.5,
                        "patience": 3,
                        "min_lr": 1e-6,
                        "verbose": True
                    },
                    "optimizer": {
                        **config.get("optimizer", {}),
                        "lr": current_lr * 0.5  # Slightly reduce learning rate for fine-tuning
                    }
                },
                "reasoning": "Adding learning rate scheduler and slightly reducing learning rate for more stable fine-tuning."
            }
        
        # If batch size is large, reduce it for better gradient estimates
        if batch_size > 64:
            return {
                "type": "configuration",
                "changes": {
                    "batch_size": max(32, batch_size // 2)
                },
                "reasoning": f"Reducing batch size from {batch_size} to {max(32, batch_size // 2)} for better gradient estimates during fine-tuning."
            }
        
        # If learning rate is still high, reduce it further
        if current_lr > 1e-4:
            return {
                "type": "configuration",
                "changes": {
                    "optimizer": {
                        **config.get("optimizer", {}),
                        "lr": max(1e-5, current_lr * 0.2)  # Reduce learning rate significantly
                    }
                },
                "reasoning": f"Reducing learning rate from {current_lr:.2e} to {max(1e-5, current_lr * 0.2):.2e} for more precise parameter updates during fine-tuning."
            }
        
        # As a last resort, try a different optimizer
        current_optimizer = config.get("optimizer", {}).get("type", "Adam").lower()
        
        # Choose optimizer based on current one
        if "adam" in current_optimizer:
            new_optimizer = "SGD"
            lr = 0.01
            momentum = 0.9
        else:
            new_optimizer = "AdamW"
            lr = 1e-4
            momentum = None
        
        return {
            "type": "configuration",
            "changes": {
                "optimizer": {
                    "type": new_optimizer,
                    "lr": lr,
                    **({"momentum": momentum} if momentum is not None else {})
                }
            },
            "reasoning": f"Switching to {new_optimizer} optimizer with learning rate {lr} for better fine-tuning behavior."
        }