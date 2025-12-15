"""
Simple test to verify PSAAttention correctness against PyTorch SDPA.
Tests with mask_ratios set to all 1 (full attention).
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psa_triton import PSAAttention, PSAConfig


def test_psa_vs_sdpa():
    """Test PSAAttention against PyTorch SDPA with full attention (all mask = 1)."""
    print("=" * 60)
    print("Testing PSAAttention vs SDPA (full attention)")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 512  # Must be divisible by block_m and block_n
    head_dim = 64

    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"dtype: {q.dtype}")

    # Config with full attention (all mask = 1)
    config = PSAConfig(
        block_m=128,
        block_n=64,
        tile_n=32,
        mask_ratios={
            1: (0.0, 1.0),  # 100% full attention
            2: (1.0, 1.0),  # 0%
            4: (1.0, 1.0),  # 0%
            8: (1.0, 1.0),  # 0%
            0: (1.0, 1.0),  # 0% skip
        },
        mask_mode='topk',
        attn_impl='new_mask_type',
        use_sim_mask=False,
    )

    # Create PSA attention
    psa = PSAAttention(config)

    # Compute PSA output
    print("\nRunning PSAAttention...")
    with torch.no_grad():
        psa_out = psa(q, k, v)

    # Compute SDPA output
    print("Running PyTorch SDPA...")
    with torch.no_grad():
        sdpa_out = F.scaled_dot_product_attention(q, k, v)

    # Compare outputs
    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)

    # Calculate differences
    abs_diff = (psa_out - sdpa_out).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Relative error
    rel_diff = abs_diff / (sdpa_out.abs() + 1e-6)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Mean relative difference: {mean_rel_diff:.6e}")

    # Check if outputs are close
    # For float16, tolerance should be higher
    atol = 1e-2
    rtol = 1e-2
    is_close = torch.allclose(psa_out, sdpa_out, atol=atol, rtol=rtol)

    print(f"\nOutputs close (atol={atol}, rtol={rtol}): {is_close}")

    if is_close:
        print("\n✅ TEST PASSED: PSAAttention matches SDPA!")
    else:
        print("\n❌ TEST FAILED: PSAAttention does not match SDPA")

        # Show sample values for debugging
        print("\nSample values (first head, first 5 positions):")
        print(f"PSA:  {psa_out[0, 0, :5, :3]}")
        print(f"SDPA: {sdpa_out[0, 0, :5, :3]}")

    return is_close


def test_different_seq_lengths():
    """Test with different sequence lengths."""
    print("\n" + "=" * 60)
    print("Testing different sequence lengths")
    print("=" * 60)

    seq_lengths = [256, 512, 1024]
    results = []

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        torch.manual_seed(42)
        q = torch.randn(1, 4, seq_len, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 4, seq_len, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 4, seq_len, 64, device="cuda", dtype=torch.float16)

        config = PSAConfig(
            block_m=128,
            block_n=64,
            tile_n=32,
            mask_ratios={1: (0.0, 1.0), 2: (1.0, 1.0), 4: (1.0, 1.0), 8: (1.0, 1.0), 0: (1.0, 1.0)},
            mask_mode='topk',
            attn_impl='new_mask_type',
        )
        psa = PSAAttention(config)

        with torch.no_grad():
            psa_out = psa(q, k, v)
            sdpa_out = F.scaled_dot_product_attention(q, k, v)

        max_diff = (psa_out - sdpa_out).abs().max().item()
        is_close = torch.allclose(psa_out, sdpa_out, atol=1e-2, rtol=1e-2)

        status = "✅" if is_close else "❌"
        print(f"  {status} Max diff: {max_diff:.6e}")
        results.append(is_close)

    return all(results)


if __name__ == "__main__":
    print("PSA Attention Correctness Test")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tests")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Run tests
    test1_passed = test_psa_vs_sdpa()
    test2_passed = test_different_seq_lengths()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Basic test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Seq length test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed")
        sys.exit(1)
