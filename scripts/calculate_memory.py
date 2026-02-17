#!/usr/bin/env python3
"""Calculate optimal configuration for L40S 48GB VRAM."""


def calculate_memory_usage():
    """Calculate memory usage for different configurations."""
    # Model parameters
    vocab_size = 32000
    d_model = 2048  # 1B parameter model
    num_layers = 24
    d_state = 64
    d_conv = 4
    expand = 2

    # Calculate parameters
    embedding_params = vocab_size * d_model
    mamba_params_per_layer = (
        d_model * d_model * expand + d_model * d_state + d_model * d_conv
    )
    mamba_params = num_layers * mamba_params_per_layer
    total_params = embedding_params + mamba_params  # weight tying

    print("=== 1B Parameter Model ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (FP16): {total_params * 2 / 1e9:.1f} GB")
    print(f"Model size (FP32): {total_params * 4 / 1e9:.1f} GB")

    # Memory usage for different batch sizes
    seq_length = 512

    print("\n=== Memory Usage for Different Batch Sizes (FP16) ===")
    for batch_size in [8, 16, 32, 64]:
        # Model memory (FP16)
        model_memory = total_params * 2 / 1e9

        # Activation memory (approximate)
        # Hidden states: batch * seq * d_model * 2 bytes (FP16)
        hidden_memory = batch_size * seq_length * d_model * 2 / 1e9

        # Gradient memory (same as model)
        gradient_memory = model_memory

        # Optimizer states (AdamW: 2x model size for FP16)
        optimizer_memory = model_memory * 2

        total_memory = model_memory + hidden_memory + gradient_memory + optimizer_memory

        print(f"Batch size {batch_size}:")
        print(f"  Model: {model_memory:.1f} GB")
        print(f"  Activations: {hidden_memory:.1f} GB")
        print(f"  Gradients: {gradient_memory:.1f} GB")
        print(f"  Optimizer: {optimizer_memory:.1f} GB")
        print(f"  Total: {total_memory:.1f} GB")
        print(f"  Headroom: {48 - total_memory:.1f} GB")
        print()

    # Check if we can increase model size
    print("=== Can We Increase Model Size? ===")

    # Try 1.5B parameters
    d_model_1_5b = 2560
    num_layers_1_5b = 28

    embedding_params_1_5b = vocab_size * d_model_1_5b
    mamba_params_per_layer_1_5b = (
        d_model_1_5b * d_model_1_5b * expand
        + d_model_1_5b * d_state
        + d_model_1_5b * d_conv
    )
    mamba_params_1_5b = num_layers_1_5b * mamba_params_per_layer_1_5b
    total_params_1_5b = embedding_params_1_5b + mamba_params_1_5b

    print(f"1.5B model parameters: {total_params_1_5b:,}")
    print(f"1.5B model size (FP16): {total_params_1_5b * 2 / 1e9:.1f} GB")

    # Memory with batch 16
    batch_size = 16
    model_memory_1_5b = total_params_1_5b * 2 / 1e9
    hidden_memory_1_5b = batch_size * seq_length * d_model_1_5b * 2 / 1e9
    total_memory_1_5b = (
        model_memory_1_5b * 4 + hidden_memory_1_5b
    )  # model + grad + optimizer * 2

    print(f"1.5B model with batch {batch_size}: {total_memory_1_5b:.1f} GB")
    print(f"Headroom: {48 - total_memory_1_5b:.1f} GB")


if __name__ == "__main__":
    calculate_memory_usage()
