"""
Model Pruning - structured and unstructured pruning for YOLO models.
Supports magnitude-based, gradient-based, and channel pruning.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class ModelPruner:
    """
    Model pruning toolkit for YOLO models.

    Supports:
    - Magnitude pruning (L1/L2 structured/unstructured)
    - Channel pruning (filter-level)
    - Slim pruning (BN-based)
    - Gradual pruning schedule

    Usage:
        pruner = ModelPruner(model)
        pruner.prune(amount=0.3, method="magnitude")
        pruner.export_pruned("pruned.pt")
    """

    def __init__(self, model: nn.Module, device: str = "auto"):
        self.model = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._pruning_masks: Dict[str, torch.Tensor] = {}

    def prune(
        self,
        amount: float = 0.3,
        method: str = "magnitude",
        exclude_layers: Optional[List[str]] = None,
        global_pruning: bool = False,
    ) -> nn.Module:
        """
        Apply pruning to the model.

        Args:
            amount: Pruning ratio (0.0 to 1.0)
            method: Pruning method - 'magnitude', 'random', 'bn_scale', 'structured'
            exclude_layers: Layer name patterns to exclude from pruning
            global_pruning: Apply global pruning across all layers

        Returns:
            Pruned model
        """
        exclude_layers = exclude_layers or ["detect", "head", "classifier"]

        # Collect prunable parameters
        parameters = []
        for name, module in self.model.named_modules():
            if any(ex in name for ex in exclude_layers):
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters.append((module, "weight"))

        if not parameters:
            print("⚠️ No prunable parameters found")
            return self.model

        if method == "magnitude":
            self._magnitude_prune(parameters, amount, global_pruning)
        elif method == "random":
            self._random_prune(parameters, amount, global_pruning)
        elif method == "bn_scale":
            self._bn_scale_prune(amount, exclude_layers)
        elif method == "structured":
            self._structured_prune(parameters, amount)
        else:
            raise ValueError(f"Unknown pruning method: {method}")

        print(f"✅ Pruning complete: {method}, amount={amount}")
        self._print_sparsity()
        return self.model

    def _magnitude_prune(self, parameters, amount, global_pruning):
        """L1 magnitude pruning."""
        if global_pruning:
            prune.global_unstructured(
                parameters,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        else:
            for module, param_name in parameters:
                prune.l1_unstructured(module, param_name, amount=amount)

    def _random_prune(self, parameters, amount, global_pruning):
        """Random pruning."""
        for module, param_name in parameters:
            prune.random_unstructured(module, param_name, amount=amount)

    def _bn_scale_prune(self, amount, exclude_layers):
        """
        BN-scale pruning: prune channels based on BN gamma values.
        Effective for structured pruning without fine-tuning.
        """
        bn_layers = []
        for name, module in self.model.named_modules():
            if any(ex in name for ex in exclude_layers):
                continue
            if isinstance(module, nn.BatchNorm2d):
                bn_layers.append((name, module))

        if not bn_layers:
            print("⚠️ No BatchNorm layers found for BN-scale pruning")
            return

        # Collect all BN gamma values
        all_gammas = []
        for name, bn in bn_layers:
            all_gammas.append(bn.weight.data.abs())

        all_gammas_cat = torch.cat(all_gammas)
        threshold = torch.quantile(all_gammas_cat, amount)

        # Zero out channels below threshold
        pruned_channels = 0
        for name, bn in bn_layers:
            mask = bn.weight.data.abs() > threshold
            bn.weight.data *= mask.float()
            bn.bias.data *= mask.float()
            pruned_channels += (~mask).sum().item()

        print(f"  Pruned {pruned_channels} channels based on BN scale")

    def _structured_prune(self, parameters, amount):
        """Structured (filter-level) pruning."""
        for module, param_name in parameters:
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, param_name, amount=amount, n=1, dim=0)

    def remove_pruning_reparam(self) -> nn.Module:
        """Remove pruning masks and make pruning permanent."""
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    pass
                if hasattr(module, "bias") and module.bias is not None:
                    try:
                        prune.remove(module, "bias")
                    except ValueError:
                        pass
        return self.model

    def fine_tune(
        self,
        train_fn,
        epochs: int = 50,
        lr: float = 0.001,
        **kwargs,
    ) -> Any:
        """
        Fine-tune the pruned model.

        Args:
            train_fn: Training function(model, epochs, lr, ...) -> results
            epochs: Fine-tuning epochs
            lr: Learning rate (typically lower than original)
        """
        print(f"🔧 Fine-tuning pruned model for {epochs} epochs, lr={lr}")
        return train_fn(self.model, epochs=epochs, lr=lr, **kwargs)

    def _print_sparsity(self):
        """Print model sparsity statistics."""
        total_params = 0
        zero_params = 0
        for name, param in self.model.named_parameters():
            if "weight" in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        if total_params > 0:
            sparsity = zero_params / total_params * 100
            print(f"  Model sparsity: {sparsity:.1f}% ({zero_params}/{total_params} zeros)")

    def export_pruned(self, save_path: Union[str, Path]) -> str:
        """Save the pruned model."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model = self.remove_pruning_reparam()
        torch.save(model.state_dict(), save_path)
        print(f"💾 Pruned model saved to: {save_path}")
        return str(save_path)

    @staticmethod
    def load_pruned(model: nn.Module, weights_path: Union[str, Path]) -> nn.Module:
        """Load pruned model weights."""
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def gradual_prune(
        self,
        initial_amount: float = 0.0,
        final_amount: float = 0.5,
        steps: int = 10,
        train_step_fn=None,
    ) -> nn.Module:
        """
        Gradual pruning schedule - increase sparsity over training steps.

        Args:
            initial_amount: Starting sparsity
            final_amount: Target sparsity
            steps: Number of pruning steps
            train_step_fn: Function to call between pruning steps
        """
        for step in range(steps):
            # Cubic schedule
            current_amount = initial_amount + (final_amount - initial_amount) * (step / steps) ** 3
            self.prune(amount=current_amount, method="magnitude")
            if train_step_fn:
                train_step_fn(step)
            print(f"  Step {step+1}/{steps}: sparsity={current_amount:.3f}")

        return self.model
