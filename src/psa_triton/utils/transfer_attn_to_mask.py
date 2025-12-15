import torch
import math
from typing import Dict, Tuple, Optional

    
def transfer_attn_to_mask(
    attn: torch.Tensor,
    mask_ratios: Optional[Dict[int, Tuple[float, float]]] = None,
    text_length: int = 226,
    mode: str = "topk",
    min_full_attn_ratio: float = 0.06,
    blocksize=32,
    compute_tile=32
) -> torch.Tensor:
    """
    Convert attention weights to multi-level pooling mask matrix.

    Args:
        attn (torch.Tensor): Attention weight matrix, shape [batch, head, seq, seq]
        mask_ratios (dict): Mask value to percentage range mapping, format {mask_value: (start_ratio, end_ratio)}
                           Default is {1: (0.0, 0.05), 2: (0.05, 0.15), 4: (0.15, 0.55), 8: (0.55, 1.0)}
                           Other positions have mask=0 (skip)
        text_length (int): Text sequence length, used to calculate special token positions
        mode (str): Mask generation mode, 'topk' or 'thresholdbound'
                   - 'topk': Generate mask based on sorted position range
                   - 'thresholdbound': Generate mask based on cumulative energy percentage
                   - 'topk_newtype': topk new format mask
                   - 'thresholdbound_newtype': thresholdbound new format mask
        min_full_attn_ratio (float): Minimum interval ratio when mask_value=1, default 0.05 (5%)
                                     Ensures full attention interval occupies at least this ratio

    Returns:
        torch.Tensor: Multi-level mask matrix, same shape as input
        - 0: skip (no attention computation)
        - 1: full attention (default top 5%)
        - 2: 2x pooling (default 5%-15%)
        - 4: 4x pooling (default 15%-55%)
        - 8: 8x pooling (default 55%-100%)
    """
    def process_text_tokens(text_length,blocksize,mask):
        if(text_length==0):
            return mask
        text_blocks=(text_length+blocksize-1)//blocksize
        mask[...,-text_blocks:]=1
        return mask
        
        
    if mask_ratios is None:
        raise ValueError("mask_ratios must be provided")

    batch, heads, seq_q, seq_k = attn.shape
    device = attn.device
    if mode == "topk":
        mask = torch.zeros_like(attn, dtype=torch.int32)
        # Original TopK mode: batch process attention weight sorting for all query positions
        # attn shape: [batch, heads, seq_q, seq_k]
        sorted_weights, indices = torch.sort(attn, dim=-1, descending=True)

        # Maintain the end_idx of previous interval
        last_end_idx = 0

        # Set corresponding range for each mask value - batch processing
        for mask_value, (start_ratio, end_ratio) in mask_ratios.items():
            # Calculate alignment multiple (mask_value=0 means skip, no alignment needed)
            if mask_value == 0:
                alignment_multiple = 1  # no alignment
            else:
                alignment_multiple = math.ceil(mask_value / (blocksize // compute_tile))

            # start_idx must equal the end_idx of previous interval
            start_idx = last_end_idx
            initial_end_idx = min(seq_k, int(seq_k * end_ratio))

            # Align interval length to multiple of alignment_multiple
            # Interval is left-closed right-open [start_idx, end_idx)
            interval_length = initial_end_idx - start_idx
            if alignment_multiple > 1 and interval_length > 0:
                remainder = interval_length % alignment_multiple
                if remainder != 0:
                    # Round up alignment: increase interval length to next multiple
                    aligned_length = interval_length + (alignment_multiple - remainder)
                else:
                    aligned_length = interval_length

                # Ensure end_idx does not exceed seq_k
                end_idx = min(start_idx + aligned_length, seq_k)

                # If end_idx reaches boundary, may need to round down alignment
                if end_idx == seq_k:
                    actual_length = end_idx - start_idx
                    aligned_length = (
                        actual_length // alignment_multiple
                    ) * alignment_multiple
                    end_idx = start_idx + aligned_length
            else:
                end_idx = initial_end_idx

            if start_idx < end_idx:
                # Create position range mask [seq_k] -> [1, 1, 1, seq_k]
                position_range = torch.arange(seq_k, device=device)
                range_mask = (position_range >= start_idx) & (position_range < end_idx)
                range_mask = range_mask.view(1, 1, 1, seq_k).expand(
                    batch, heads, seq_q, -1
                )

                # Batch set mask values - operate on all query positions simultaneously
                # Ensure mask_value type matches mask
                mask_value_tensor = torch.tensor(
                    mask_value, dtype=mask.dtype, device=device
                )
                mask.scatter_(
                    -1,
                    indices,
                    torch.where(
                        range_mask, mask_value_tensor, mask.gather(-1, indices)
                    ),
                )

                # Update last_end_idx
                last_end_idx = end_idx

        # Process text tokens: set text blocks to full attention
        mask = process_text_tokens(text_length, blocksize, mask)

    elif mode == "thresholdbound":
        mask = torch.zeros_like(attn, dtype=torch.int32)
        # Energy bound mode: generate mask based on cumulative energy percentage
        # Each row (each query) is processed independently, determine interval and align based on energy accumulation ratio
        # attn shape: [batch, heads, seq, seq]
        sorted_weights, indices = torch.sort(attn, dim=-1, descending=True)

        # Calculate cumulative energy for each row (cumulative sum) [batch, heads, seq, seq]
        #row_sums = attn.sum(dim=-1, keepdim=True)  # [batch, heads, seq, 1]
        cumsum_weights = torch.cumsum(
            sorted_weights, dim=-1
        )  # [batch, heads, seq, seq]

        # Calculate cumulative energy ratio [batch, heads, seq, seq]
        energy_ratio = cumsum_weights / (cumsum_weights[...,-1:])

        # Process in ascending order of end_ratio (ensure interval continuity)
        sorted_mask_items = sorted(mask_ratios.items(), key=lambda x: x[1][1])

        # Process each row independently, maintain last_aligned_end_idx for each row
        # last_aligned_end_indices shape: [batch, heads, seq_q]
        last_aligned_end_indices = torch.zeros(
            (batch, heads, seq_q), dtype=torch.long, device=device
        )

        for mask_value, (config_start_ratio, config_end_ratio) in sorted_mask_items:
            
            alignment_multiple = math.ceil(mask_value / (blocksize // compute_tile))

            # Find the last position in each row that satisfies energy_ratio <= end_ratio
            # valid_mask shape: [batch, heads, seq_q, seq_k]
            valid_mask = energy_ratio <= (config_end_ratio + 1e-6)

            # # Find the last True position in each row (on seq_k dimension)
            # # Flip valid_mask, find first True, then calculate original position
            # flipped_mask = torch.flip(
            #     valid_mask, dims=[-1]
            # )  # [batch, heads, seq_q, seq_k]
            # first_true_in_flipped = torch.argmax(
            #     flipped_mask.to(torch.long), dim=-1
            # )  # [batch, heads, seq_q]

            # # Convert back to original index: seq_k - 1 - first_true_in_flipped
            # # But need to handle the case where there is no True (argmax will return 0)
            has_valid = valid_mask.any(dim=-1)  # [batch, heads, seq_q]
            # last_valid_idx = seq_k - 1 - first_true_in_flipped  # [batch, heads, seq_q]
            last_valid_idx = valid_mask.sum(dim=-1)
            # initial_end_idx = last_valid_idx + 1 (include this position)
            initial_end_indices = last_valid_idx + 1  # [batch, heads, seq_q]

            # For rows without valid positions, use actual_start_idx
            initial_end_indices = torch.where(
                has_valid, initial_end_indices, last_aligned_end_indices
            )

            # The start of current interval is the aligned end_index of previous interval
            actual_start_indices = last_aligned_end_indices  # [batch, heads, seq_q]

            # Align interval length to multiple of alignment_multiple
            # Interval is left-closed right-open [actual_start_indices, aligned_end_indices)
            if alignment_multiple > 1:
                # Calculate current interval length
                interval_lengths = (
                    initial_end_indices - actual_start_indices
                ).clamp(min=0)  # [batch, heads, seq_q]

                # Calculate remainder
                remainder = interval_lengths % alignment_multiple

                # Round up alignment for interval length
                aligned_lengths = torch.where(
                    remainder != 0,
                    interval_lengths + (alignment_multiple - remainder),
                    interval_lengths,
                )

                # Calculate aligned end_idx
                aligned_end_indices = actual_start_indices + aligned_lengths

                # Ensure not exceeding seq_k
                aligned_end_indices = torch.clamp(aligned_end_indices, max=seq_k)

                # If reaching boundary, round down alignment
                at_boundary = aligned_end_indices == seq_k
                actual_lengths = aligned_end_indices - actual_start_indices
                downaligned_lengths = (
                    actual_lengths // alignment_multiple
                ) * alignment_multiple
                aligned_end_indices = torch.where(
                    at_boundary,
                    actual_start_indices + downaligned_lengths,
                    aligned_end_indices,
                )
            else:
                aligned_end_indices = torch.clamp(
                    initial_end_indices, min=0, max=seq_k
                )
                aligned_end_indices = torch.maximum(
                    aligned_end_indices, actual_start_indices
                )
                if mask_value == 1:
                    min_index = min(seq_k, int(seq_k * min_full_attn_ratio) + 1)
                    min_index_tensor = torch.tensor(
                        min_index, dtype=torch.long, device=device
                    )
                    aligned_end_indices = torch.maximum(
                        aligned_end_indices, min_index_tensor
                    )
            # Set mask value for each valid interval
            # Create position index [seq_k]
            position_range = torch.arange(seq_k, device=device)  # [seq_k]

            # Expand dimensions for broadcasting: [1, 1, 1, seq_k]
            position_range = position_range.view(1, 1, 1, seq_k)

            # Expand start and end for broadcasting: [batch, heads, seq_q, 1]
            actual_start_expanded = actual_start_indices.unsqueeze(
                -1
            )  # [batch, heads, seq_q, 1]
            aligned_end_expanded = aligned_end_indices.unsqueeze(
                -1
            )  # [batch, heads, seq_q, 1]

            # Create range mask: [batch, heads, seq_q, seq_k]
            range_mask = (position_range >= actual_start_expanded) & (
                position_range < aligned_end_expanded
            )

            # Use scatter to set mask value
            # Ensure mask_value type matches mask
            mask_value_tensor = torch.tensor(
                mask_value, dtype=mask.dtype, device=device
            )
            mask.scatter_(
                -1,
                indices,
                torch.where(range_mask, mask_value_tensor, mask.gather(-1, indices)),
            )

            # Update last_aligned_end_indices
            last_aligned_end_indices = aligned_end_indices
            start_idx = 0

        # Process text tokens: set text blocks to full attention
        mask = process_text_tokens(text_length, blocksize, mask)

    elif mode == "topk_newtype":
        attn_shape = list(attn.shape)
        attn_shape[-1] = attn_shape[-1] + len(mask_ratios)
        attn_shape = tuple(attn_shape)
        mask = torch.zeros(attn_shape, dtype=torch.int32, device=device)
        sorted_weights, indices = torch.sort(attn, dim=-1, descending=True)
        counter = 0
        # Maintain the end_idx of previous interval
        last_end_idx = 0

        # Set corresponding range for each mask value - batch processing
        for mask_value, (start_ratio, end_ratio) in mask_ratios.items():
            # Calculate alignment multiple (mask_value=0 means skip, no alignment needed)
            if mask_value == 0:
                alignment_multiple = 1  # no alignment
            else:
                alignment_multiple = math.ceil(mask_value / (blocksize // compute_tile))

            # start_idx must equal the end_idx of previous interval
            start_idx = last_end_idx
            initial_end_idx = min(seq_k, int(seq_k * end_ratio))

            # Align interval length to multiple of alignment_multiple
            # Interval is left-closed right-open [start_idx, end_idx)
            interval_length = initial_end_idx - start_idx
            if alignment_multiple > 1 and interval_length > 0:
                remainder = interval_length % alignment_multiple
                if remainder != 0:
                    # Round up alignment: increase interval length to next multiple
                    aligned_length = interval_length + (alignment_multiple - remainder)
                else:
                    aligned_length = interval_length

                # Ensure end_idx does not exceed seq_k
                end_idx = min(start_idx + aligned_length, seq_k)

                # If end_idx reaches boundary, may need to round down alignment
                if end_idx == seq_k:
                    actual_length = end_idx - start_idx
                    aligned_length = (
                        actual_length // alignment_multiple
                    ) * alignment_multiple
                    end_idx = start_idx + aligned_length
            else:
                end_idx = initial_end_idx

            # Even if start_idx >= end_idx, maintain mask format consistency
            if start_idx < end_idx:
                # Create position range mask [seq_k] -> [1, 1, 1, seq_k]
                position_range = torch.arange(seq_k, device=device)
                range_mask = (position_range >= start_idx) & (position_range < end_idx)
                range_mask = range_mask.view(1, 1, 1, seq_k).expand(
                    batch, heads, seq_q, -1
                )

                # Use scatter to set mask value
                mask[:, :, :, counter : counter + seq_k] = torch.where(
                    range_mask, indices, mask[:, :, :, counter : counter + seq_k]
                )
            else:
                end_idx = start_idx  # Keep unchanged
            # Set separator -1 at end_idx position (must set regardless of whether interval is valid)
            # Create a scalar tensor representing end_idx position
            aligned_end_idx_tensor = torch.full(
                (batch, heads, seq_q, 1),
                counter + end_idx,
                dtype=torch.long,
                device=device,
            )

            # Create value tensor
            values = torch.full(
                (batch, heads, seq_q, 1), -1, dtype=mask.dtype, device=device
            )

            # Use scatter_ to set -1 at specified position
            mask.scatter_(dim=-1, index=aligned_end_idx_tensor, src=values)

            # Update last_end_idx and counter (must update regardless of whether interval is valid)
            last_end_idx = end_idx
            counter += 1
    elif mode == "thresholdbound_newtype":
        # Energy bound mode: generate mask based on cumulative energy percentage
        # Each row (each query) is processed independently, determine interval and align based on energy accumulation ratio
        # attn shape: [batch, heads, seq_q, seq_k] mask types 0 1 2 4 8 16 require 5 separators -1
        attn_shape = list(attn.shape)
        attn_shape[-1] = attn_shape[-1] + len(mask_ratios)
        attn_shape = tuple(attn_shape)
        mask = torch.zeros(attn_shape, dtype=torch.int32, device=device)
        sorted_weights, indices = torch.sort(attn, dim=-1, descending=True)

        # Calculate cumulative energy for each row (cumulative sum) [batch, heads, seq_q, seq_k]
        # row_sums = attn.sum(dim=-1, keepdim=True)  # [batch, heads, seq_q, 1]
        cumsum_weights = torch.cumsum(
            sorted_weights, dim=-1
        )  # [batch, heads, seq_q, seq_k]

        # Calculate cumulative energy ratio [batch, heads, seq_q, seq_k]
        energy_ratio = cumsum_weights / (cumsum_weights[...,-1:])

        # Process in ascending order of end_ratio (ensure interval continuity)
        sorted_mask_items = sorted(mask_ratios.items(), key=lambda x: x[1][1])

        # Process each row independently, maintain last_aligned_end_idx for each row
        # last_aligned_end_indices shape: [batch, heads, seq_q]
        last_aligned_end_indices = torch.zeros(
            (batch, heads, seq_q), dtype=torch.long, device=device
        )
        counter = 0
        for mask_value, (config_start_ratio, config_end_ratio) in sorted_mask_items:
            # Calculate alignment multiple: ceil(mask_value / (blocksize // compute_tile))
            alignment_multiple = math.ceil(mask_value / (blocksize // compute_tile))

            # Find the last position in each row that satisfies energy_ratio <= end_ratio
            # valid_mask shape: [batch, heads, seq_q, seq_k]
            valid_mask = energy_ratio <= (config_end_ratio + 1e-6)
            # Find the last True position in each row (on seq_k dimension)
            # Flip valid_mask, find first True, then calculate original position
            # flipped_mask = torch.flip(
            #     valid_mask, dims=[-1]
            # )  # [batch, heads, seq_q, seq_k]
            # first_true_in_flipped = torch.argmax(
            #     flipped_mask.to(torch.long), dim=-1
            # )  # [batch, heads, seq_q]

            # Convert back to original index: seq_k - 1 - first_true_in_flipped
            # But need to handle the case where there is no True (argmax will return 0)
            has_valid = valid_mask.any(dim=-1)  # [batch, heads, seq_q]
            last_valid_idx = valid_mask.sum(dim=-1)

            # initial_end_idx = last_valid_idx + 1 (include this position)
            initial_end_indices = last_valid_idx + 1  # [batch, heads, seq_q]

            # For rows without valid positions, use actual_start_idx
            initial_end_indices = torch.where(
                has_valid, initial_end_indices, last_aligned_end_indices
            )

            # The start of current interval is the aligned end_index of previous interval
            actual_start_indices = last_aligned_end_indices  # [batch, heads, seq_q]

            # Align interval length to multiple of alignment_multiple
            # Interval is left-closed right-open [actual_start_indices, aligned_end_indices)
            if alignment_multiple > 1:
                # Calculate current interval length
                interval_lengths = (
                    initial_end_indices - actual_start_indices
                ).clamp(min=0)  # [batch, heads, seq_q]

                # Calculate remainder
                remainder = interval_lengths % alignment_multiple

                # Round up alignment for interval length
                aligned_lengths = torch.where(
                    remainder != 0,
                    interval_lengths + (alignment_multiple - remainder),
                    interval_lengths,
                )

                # Calculate aligned end_idx
                aligned_end_indices = actual_start_indices + aligned_lengths

                # Ensure not exceeding seq_k
                aligned_end_indices = torch.clamp(aligned_end_indices, max=seq_k)

                # If reaching boundary, round down alignment
                at_boundary = aligned_end_indices == seq_k
                actual_lengths = aligned_end_indices - actual_start_indices
                downaligned_lengths = (
                    actual_lengths // alignment_multiple
                ) * alignment_multiple
                aligned_end_indices = torch.where(
                    at_boundary,
                    actual_start_indices + downaligned_lengths,
                    aligned_end_indices,
                )
            else:
                aligned_end_indices = torch.clamp(
                    initial_end_indices, min=0, max=seq_k
                )
                aligned_end_indices = torch.maximum(
                    aligned_end_indices, actual_start_indices
                )
                if mask_value == 1:
                    min_index = min(seq_k, int(seq_k * min_full_attn_ratio) + 1)
                    min_index_tensor = torch.tensor(
                        min_index, dtype=torch.long, device=device
                    )
                    aligned_end_indices = torch.maximum(
                        aligned_end_indices, min_index_tensor
                    )

            # Set mask value for each interval (must process regardless of whether interval is valid, to maintain format consistency)
            # Create position index [seq_k]
            position_range = torch.arange(seq_k, device=device)  # [seq_k]

            # Expand dimensions for broadcasting: [1, 1, 1, seq_k]
            position_range = position_range.view(1, 1, 1, seq_k)

            # Expand start and end for broadcasting: [batch, heads, seq_q, 1]
            actual_start_expanded = actual_start_indices.unsqueeze(
                -1
            )  # [batch, heads, seq_q, 1]
            aligned_end_expanded = aligned_end_indices.unsqueeze(
                -1
            )  # [batch, heads, seq_q, 1]

            # Create range mask: [batch, heads, seq_q, seq_k]
            range_mask = (position_range >= actual_start_expanded) & (
                position_range < aligned_end_expanded
            )

            # Use scatter to set mask value (range_mask will automatically handle invalid intervals)
            mask[:, :, :, counter : counter + seq_k] = torch.where(
                range_mask, indices, mask[:, :, :, counter : counter + seq_k]
            )

            # Set separator -1 at end_idx position (must set regardless of whether interval is valid)
            values = torch.full_like(
                aligned_end_expanded, -1, dtype=mask.dtype, device=device
            )
            separator_index = torch.clamp(
                aligned_end_expanded + counter, min=0, max=mask.size(-1) - 1
            )
            mask.scatter_(dim=-1, index=separator_index, src=values)

            # Update last_aligned_end_indices and counter (must update regardless of whether interval is valid)
            last_aligned_end_indices = aligned_end_indices
            start_idx = 0
            counter = 1 + counter

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'topk' or 'thresholdbound'")

    #TODO: Add special token processing logic

    return mask


def calc_density(mask):
    """
    Calculate average computation density for mask (old format mask)

    Args:
        mask: Mask tensor with values 0, 1, 2, 4, 8, etc.
              - 0: computation amount is 0
              - 1: computation amount is 1 (full attention)
              - 2: computation amount is 0.5 (2x pooling)
              - 4: computation amount is 0.25 (4x pooling)
              - 8: computation amount is 0.125 (8x pooling)

    Returns:
        float: Average computation density
        list: Density value list for each head
    """
    # Create computation amount mapping
    density_map = torch.zeros_like(mask, dtype=torch.float32)

    # For non-zero values, computation amount is 1/mask_value
    non_zero_mask = mask > 0
    density_map[non_zero_mask] = 1.0 / mask[non_zero_mask].float()

    # Calculate average density
    avg_density = density_map.mean().item()

    # Calculate density value for each head [batch, heads, seq, seq] -> [heads]
    # Average for each head
    batch_size, num_heads, seq_len, _ = density_map.shape
    per_head_density = []
    for h in range(num_heads):
        head_density = density_map[:, h, :, :].mean().item()
        per_head_density.append(head_density)

    return avg_density, per_head_density


def calc_density_newtype(mask):
    """
    Calculate average computation density for new format mask

    New format mask structure: [batch, heads, seq_q, seq_k + num_segments]
    - Use -1 as segment separator
    - Segment 1: pooling factor = 1 (full attention)
    - Segment 2: pooling factor = 2 (2x pooling)
    - Segment 3: pooling factor = 4 (4x pooling)
    - Segment 4: pooling factor = 8 (8x pooling)
    - Segment 5: pooling factor = 0 (skip, no computation)

    Density calculation formula:
    For each row (each query):
    density = (len1*1 + len2/2 + len3/4 + len4/8) / total valid elements
    where total valid elements = mask.size(-1) - 5 (remove 5 separators -1)

    Args:
        mask: New format mask tensor, contains indices and separators (-1)

    Returns:
        float: Average computation density
        list: Density value list for each head
    """
    batch_size, num_heads, seq_q, extended_seq = mask.shape

    # Define pooling factor order (fixed order: 1, 2, 4, 8, 0)
    pooling_factors = [1, 2, 4, 8, 0]
    num_segments = len(pooling_factors)

    # Total valid elements (remove 5 separators -1)
    total_valid_elements = extended_seq - num_segments

    # Calculate density for each row [batch, heads, seq_q]
    density_per_query = torch.zeros((batch_size, num_heads, seq_q),
                                     dtype=torch.float32, device=mask.device)

    # For each segment, calculate its length and contribution
    for segment_idx, pooling_factor in enumerate(pooling_factors):
        # Find start and end positions of current segment
        # Each segment starts after segment_idx separators
        # Ends before next -1

        # Find positions of -1 in entire sequence [batch, heads, seq_q, extended_seq]
        is_separator = (mask == -1)

        # For each row, find positions of segment_idx-th and (segment_idx+1)-th -1
        # Calculate position of segment_idx-th -1 in each row
        separator_positions = is_separator.cumsum(dim=-1)  # [batch, heads, seq_q, extended_seq]

        # Create mask for current segment: separator_positions == segment_idx means not yet reached next separator
        in_current_segment = (separator_positions == segment_idx) & (mask != -1)

        # Count valid elements in current segment for each row [batch, heads, seq_q]
        segment_lengths = in_current_segment.sum(dim=-1).float()

        # If pooling factor is 0 (skip), no density contribution
        if pooling_factor == 0:
            continue

        # Density contribution of current segment = segment_length / pooling_factor
        density_per_query += segment_lengths / pooling_factor

    # Normalization: divide by total valid elements
    density_per_query = density_per_query / total_valid_elements

    # Calculate average density (average over all dimensions)
    avg_density = density_per_query.mean().item()

    # Calculate density value for each head (average over batch and seq_q dimensions)
    per_head_density = []
    for h in range(num_heads):
        head_density = density_per_query[:, h, :].mean().item()
        per_head_density.append(head_density)

    return avg_density, per_head_density

def transfer_mask_to_new_type(mask, blocksize=32, compute_tile=32):
    """
    Convert old format mask to new format mask (vectorized version)

    Old format mask: [batch, heads, seq, seq], values are 0,1,2,4,8
    New format mask: [batch, heads, seq, seq + 5], use -1 as separator

    Args:
        mask: Old format mask tensor

    Returns:
        torch.Tensor: New format mask tensor
    """
    batch_size, num_heads, seq_len, _ = mask.shape
    device = mask.device

    # Calculate size of last dimension of new format mask
    num_segments = 5  # Order: 1, 2, 4, 8, 0
    new_seq_len = seq_len + num_segments

    # Initialize new format mask
    new_mask = torch.zeros(
        (batch_size, num_heads, seq_len, new_seq_len),
        dtype=torch.int32,
        device=device,
    )

    # Flatten to [total_rows, seq_len] for batch processing
    mask_flat = mask.view(-1, seq_len)
    new_mask_flat = new_mask.view(-1, new_seq_len)
    total_rows = mask_flat.size(0)

    # Pre-calculate alignment related constants
    divisor = max(1, blocksize // compute_tile)
    mask_values = [1, 2, 4, 8, 0]
    alignment_multiples = []
    for val in mask_values:
        if val == 0:
            alignment_multiples.append(1)
        else:
            alignment_multiples.append(max(1, math.ceil(val / divisor)))

    # Replace 0 values with 100 for sorting [total_rows, seq_len]
    mask_values_replaced = mask_flat.clone()
    zero_mask = mask_values_replaced == 0
    mask_values_replaced[zero_mask] = 100

    # Batch sort [total_rows, seq_len]
    sorted_vals, sorted_indices = torch.sort(mask_values_replaced, dim=-1)

    # Batch calculate counts for each segment [total_rows, 5]
    value_order = torch.tensor([1, 2, 4, 8, 100], dtype=mask.dtype, device=device)
    counts = torch.zeros((total_rows, num_segments), dtype=torch.long, device=device)
    for seg_idx, val in enumerate(value_order):
        counts[:, seg_idx] = (sorted_vals == val).sum(dim=-1)

    # Alignment processing - simplified version: temporarily don't use complex borrowing logic, use round down alignment
    # For each segment, calculate aligned length
    aligned_counts = counts.clone()

    # For each segment (except the last one), perform round down alignment
    for seg_idx in range(num_segments - 1):
        align_multiple = alignment_multiples[seg_idx]
        if align_multiple > 1:
            # Calculate remainder [total_rows]
            remainder = aligned_counts[:, seg_idx] % align_multiple

            # Round down alignment: reduce current segment, increase to next segment
            need_reduce = remainder > 0
            aligned_counts[:, seg_idx] = torch.where(
                need_reduce,
                aligned_counts[:, seg_idx] - remainder,
                aligned_counts[:, seg_idx]
            )
            aligned_counts[:, seg_idx + 1] = torch.where(
                need_reduce,
                aligned_counts[:, seg_idx + 1] + remainder,
                aligned_counts[:, seg_idx + 1]
            )

    # Build new format mask - batch processing
    # Calculate start position for each segment [total_rows, 5]
    segment_starts = torch.zeros((total_rows, num_segments), dtype=torch.long, device=device)
    for seg_idx in range(1, num_segments):
        segment_starts[:, seg_idx] = segment_starts[:, seg_idx - 1] + aligned_counts[:, seg_idx - 1]

    # Calculate start position for each segment in new mask (including separators)
    new_segment_starts = torch.zeros((total_rows, num_segments), dtype=torch.long, device=device)
    for seg_idx in range(num_segments):
        new_segment_starts[:, seg_idx] = aligned_counts[:, :seg_idx].sum(dim=-1) + seg_idx

    # Fill data for each segment
    for seg_idx in range(num_segments):
        seg_length_max = aligned_counts[:, seg_idx].max().item()

        if seg_length_max > 0:
            # Create position range [seg_length_max]
            pos_range = torch.arange(seg_length_max, device=device)

            # Expand dimensions [total_rows, seg_length_max]
            pos_range_expanded = pos_range.unsqueeze(0).expand(total_rows, -1)

            # Create valid mask [total_rows, seg_length_max]
            valid_mask = pos_range_expanded < aligned_counts[:, seg_idx].unsqueeze(-1)

            # Calculate source index position [total_rows, seg_length_max]
            src_idx = segment_starts[:, seg_idx].unsqueeze(-1) + pos_range_expanded
            src_idx = torch.clamp(src_idx, 0, seq_len - 1)

            # Get corresponding sorted_indices [total_rows, seg_length_max]
            segment_indices = torch.gather(sorted_indices, 1, src_idx).to(torch.int32)

            # Calculate target index position [total_rows, seg_length_max]
            dst_idx = new_segment_starts[:, seg_idx].unsqueeze(-1) + pos_range_expanded
            dst_idx = torch.clamp(dst_idx, 0, new_seq_len - 1)

            # Use scatter to fill - only fill valid positions
            # Method: selectively scatter using mask
            # For invalid positions, use original value at dst_idx (although all are 0, but keep consistency)
            masked_indices = segment_indices * valid_mask.to(torch.int32)
            new_mask_flat.scatter_(1, dst_idx, masked_indices)
        # Set separator -1
        separator_idx = (new_segment_starts[:, seg_idx] + aligned_counts[:, seg_idx]).unsqueeze(-1)
        separator_idx = torch.clamp(separator_idx, 0, new_seq_len - 1)
        new_mask_flat.scatter_(1, separator_idx, torch.full((total_rows, 1), -1, dtype=torch.int32, device=device))

    return new_mask

    
