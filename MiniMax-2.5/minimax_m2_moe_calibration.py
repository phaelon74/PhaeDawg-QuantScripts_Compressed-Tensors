"""
MiniMax-M2.5 (MiniMaxM2SparseMoeBlock) MoE Calibration Module for llmcompressor

CRITICAL: Ensures EVERY expert is activated during calibration for EVERY sample.

The original MiniMaxM2Experts.forward() only runs experts selected by the router
(expert_hit = experts that received at least one token). This means many experts
never receive calibration data, leading to poor quantization of rarely-activated experts.

This calibration module:
1. Runs ALL tokens through ALL experts during each forward pass
2. Applies routing weights for the final output (preserving correct semantics)
3. Allows AWQ/GPTQ hooks to collect statistics on every expert's w1, w2, w3 layers

Architecture: MiniMax-M2.5 uses Llama-style MoE with w1, w2, w3 (SwiGLU):
  - w1: gate projection (hidden -> intermediate), passed through act_fn
  - w3: up projection (hidden -> intermediate)
  - w2: down projection (intermediate -> hidden)
  - forward: act_fn(w1(x)) * w3(x) -> w2(...)
"""

import torch
import torch.nn as nn
from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("MiniMaxM2SparseMoeBlock")
class CalibrationMiniMaxM2SparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of MiniMaxM2SparseMoeBlock that runs ALL experts for ALL tokens.

    During calibration, every expert's w1, w2, w3 Linear layers receive forward passes
    for every token in the batch, enabling proper AWQ/GPTQ scale collection.
    """

    is_permanent = True  # Keep wrappers during quantization
    _forward_call_count = 0
    _last_log_count = 0

    def __init__(
        self,
        original: nn.Module,  # MiniMaxM2SparseMoeBlock
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.gate = original.gate
        self.experts = original.experts  # ModuleList of MiniMaxM2MLP (w1, w2, w3)
        self.register_buffer("e_score_correction_bias", original.e_score_correction_bias.clone())
        self.top_k = original.top_k
        self.jitter_noise = original.jitter_noise
        self.calibrate_all_experts = calibrate_all_experts
        self.num_experts = len(self.experts)

        print(f"[CalibrationMiniMaxM2SparseMoeBlock] Created for {self.num_experts} experts")
        print(f"  All experts will receive ALL tokens during calibration (w1, w2, w3)")

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        """Replicates routing logic from MiniMaxM2SparseMoeBlock."""
        routing_weights = torch.nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: ALL tokens through ALL experts for calibration.

        Each expert's w1, w2, w3 are invoked for every token, allowing AWQ to
        collect proper activation statistics. Final output uses routing weights.
        """
        CalibrationMiniMaxM2SparseMoeBlock._forward_call_count += 1
        if CalibrationMiniMaxM2SparseMoeBlock._forward_call_count - CalibrationMiniMaxM2SparseMoeBlock._last_log_count >= 500:
            print(f"[CalibrationMiniMaxM2SparseMoeBlock] forward() called "
                  f"{CalibrationMiniMaxM2SparseMoeBlock._forward_call_count} times")
            CalibrationMiniMaxM2SparseMoeBlock._last_log_count = CalibrationMiniMaxM2SparseMoeBlock._forward_call_count

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_experts = len(self.experts)

        # Get routing decisions
        router_logits = self.gate(hidden_states_flat)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)

        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Expert mask: [num_experts, num_tokens, top_k]
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=num_experts
        ).permute(2, 0, 1)

        # Run ALL tokens through ALL experts for calibration
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]

            # CRITICAL: Run ALL tokens through this expert - enables AWQ to collect stats
            expert_output_full = expert(hidden_states_flat)

            # Apply routing weights for final output
            mask = expert_mask[expert_idx]  # [num_tokens, top_k]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_output = expert_output_full[token_indices]
                expert_weights = top_k_weights[token_indices, weight_indices].to(expert_output.dtype)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))

        hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits
