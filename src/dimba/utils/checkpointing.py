"""Progressive checkpointing utilities for DIMBA.

Provides milestone-based checkpointing that saves models as they grow
through parameter count targets (e.g., 1B, 5B, 10B, 30B params).
"""

import os
import json
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path


class ProgressiveCheckpointManager:
    """Manages milestone-based progressive checkpointing.
    
    Tracks model parameter count and saves checkpoints when milestones are reached.
    
    Args:
        milestones: List of parameter count targets (e.g., [1e9, 5e9, 10e9])
        save_dir: Directory to save progressive checkpoints
        enabled: Whether progressive checkpointing is enabled
    """
    
    def __init__(
        self,
        milestones: List[int],
        save_dir: str = "./progressive_checkpoints",
        enabled: bool = True,
    ):
        self.milestones = sorted([int(m) for m in milestones])
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        
        # Track which milestones have been reached
        self.reached_milestones: set = set()
        self.next_milestone_idx: int = 0
        
        # Create save directory if needed
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def get_next_milestone(self) -> Optional[int]:
        """Get the next un-reached milestone.
        
        Returns:
            Next milestone parameter count, or None if all reached
        """
        while self.next_milestone_idx < len(self.milestones):
            milestone = self.milestones[self.next_milestone_idx]
            if milestone not in self.reached_milestones:
                return milestone
            self.next_milestone_idx += 1
        return None
    
    def count_parameters(self, model: torch.nn.Module) -> int:
        """Count trainable parameters in model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def should_save_checkpoint(self, model: torch.nn.Module) -> tuple[bool, Optional[int]]:
        """Check if a milestone has been reached.
        
        Args:
            model: Current model instance
            
        Returns:
            (should_save, milestone_reached) tuple
        """
        if not self.enabled:
            return False, None
        
        current_params = self.count_parameters(model)
        next_milestone = self.get_next_milestone()
        
        if next_milestone is None:
            return False, None
        
        if current_params >= next_milestone:
            self.reached_milestones.add(next_milestone)
            self.next_milestone_idx += 1
            return True, next_milestone
        
        return False, None
    
    def format_param_count(self, count: int) -> str:
        """Format parameter count in human-readable format.
        
        Args:
            count: Parameter count
            
        Returns:
            Formatted string (e.g., "1.0B", "500M")
        """
        if count >= 1e9:
            return f"{count / 1e9:.1f}B"
        elif count >= 1e6:
            return f"{count / 1e6:.1f}M"
        else:
            return f"{count / 1e3:.1f}K"
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        global_step: int,
        milestone: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a progressive checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            global_step: Current training step
            milestone: Parameter count milestone that was reached
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        if not self.enabled:
            raise RuntimeError("Progressive checkpointing is disabled")
        
        # Format milestone for filename
        milestone_str = self.format_param_count(milestone)
        
        # Create checkpoint filename
        filename = f"checkpoint_{milestone_str}_step_{global_step}.pt"
        filepath = self.save_dir / filename
        
        # Prepare checkpoint data
        current_params = self.count_parameters(model)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "global_step": global_step,
            "current_params": current_params,
            "target_milestone": milestone,
            "milestones": self.milestones,
            "reached_milestones": list(self.reached_milestones),
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        # Add metadata
        checkpoint["metadata"] = metadata or {}
        checkpoint["metadata"]["progressive_checkpoint"] = True
        checkpoint["metadata"]["milestone_str"] = milestone_str
        checkpoint["metadata"]["param_count_formatted"] = self.format_param_count(current_params)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Save metadata JSON for easy inspection
        meta_path = self.save_dir / f"checkpoint_{milestone_str}_step_{global_step}.json"
        json_meta = {
            "filename": filename,
            "global_step": global_step,
            "current_params": current_params,
            "target_milestone": milestone,
            "milestones": self.milestones,
            "reached_milestones": list(self.reached_milestones),
            "param_count_formatted": self.format_param_count(current_params),
            "milestone_str": milestone_str,
        }
        with open(meta_path, "w") as f:
            json.dump(json_meta, f, indent=2)
        
        return str(filepath)
    
    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load a progressive checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            Checkpoint metadata dict
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore milestone tracking
        self.reached_milestones = set(checkpoint.get("reached_milestones", []))
        
        # Find next milestone index
        self.next_milestone_idx = 0
        for i, m in enumerate(self.milestones):
            if m not in self.reached_milestones:
                self.next_milestone_idx = i
                break
        
        return {
            "global_step": checkpoint.get("global_step", 0),
            "current_params": checkpoint.get("current_params", 0),
            "target_milestone": checkpoint.get("target_milestone", 0),
            "metadata": checkpoint.get("metadata", {}),
        }
    
    def get_checkpoint_path_for_milestone(self, milestone: int) -> Optional[Path]:
        """Find checkpoint file for a specific milestone.
        
        Args:
            milestone: Parameter count milestone
            
        Returns:
            Path to checkpoint if found, None otherwise
        """
        milestone_str = self.format_param_count(milestone)
        pattern = f"checkpoint_{milestone_str}_step_*.pt"
        matches = list(self.save_dir.glob(pattern))
        
        if matches:
            # Return most recent (highest step number)
            return sorted(matches)[-1]
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all progressive checkpoints.
        
        Returns:
            List of checkpoint metadata dicts
        """
        checkpoints = []
        
        for json_file in self.save_dir.glob("checkpoint_*.json"):
            with open(json_file) as f:
                metadata = json.load(f)
                checkpoints.append(metadata)
        
        # Sort by milestone
        return sorted(checkpoints, key=lambda x: x["target_milestone"])
    
    def status_dict(self) -> Dict[str, Any]:
        """Get current status as a dictionary.
        
        Returns:
            Status dict with current state
        """
        return {
            "enabled": self.enabled,
            "milestones": self.milestones,
            "reached_milestones": list(self.reached_milestones),
            "next_milestone": self.get_next_milestone(),
            "num_checkpoints": len(list(self.save_dir.glob("checkpoint_*.pt"))),
            "save_dir": str(self.save_dir),
        }


def parse_milestone_input(input_str: str) -> List[int]:
    """Parse milestone input string into parameter counts.
    
    Accepts formats like:
    - "1,5,10,30" → [1e9, 5e9, 10e9, 30e9]
    - "1B,5B,10B" → [1e9, 5e9, 10e9]
    - "100M,500M,1B" → [100e6, 500e6, 1e9]
    
    Args:
        input_str: Comma-separated milestone string
        
    Returns:
        List of parameter counts
    """
    milestones = []
    
    for part in input_str.split(","):
        part = part.strip().upper()
        
        # Parse with suffix
        if part.endswith("B"):
            value = float(part[:-1]) * 1e9
        elif part.endswith("M"):
            value = float(part[:-1]) * 1e6
        elif part.endswith("K"):
            value = float(part[:-1]) * 1e3
        else:
            # Assume billions if no suffix and number is small
            value = float(part)
            if value < 1000:  # Likely in billions
                value *= 1e9
        
        milestones.append(int(value))
    
    return sorted(milestones)
