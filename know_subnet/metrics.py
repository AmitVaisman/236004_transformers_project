import torch
import torch.nn as nn
from torch.nn import functional as F
from know_subnet.lm.qwen import SelectivePrunedQwenLM
from typing import Any, Dict, Union

# Define cross-entropy loss without reduction for per-token losses.
cse_loss = nn.CrossEntropyLoss(reduction='none')

def hf_perp_func(hf_cross_entropy_loss: torch.Tensor) -> float:
    """Compute perplexity from a cross entropy loss tensor from HuggingFace."""
    return torch.exp(hf_cross_entropy_loss).item()
    
# TODO: Delete this if you don't find instances of it being used.
# def batch_loss_func(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
#     """
#     Compute the per-token loss (using only the masked tokens) for a batch.
#     """
#     with torch.no_grad():
#         assert vocab_size == logits.size(-1), "Mismatch between vocab_size and logits' last dimension."
#         logits_flat = logits.view(-1, vocab_size)
#         labels_flat = labels.view(-1)
#         mask_flat = mask.view(-1)
#         loss = cse_loss(logits_flat, labels_flat)
#         return loss[mask_flat]

def cse_batch_loss_func(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, lm: str) -> torch.Tensor:
    """
    Compute the cross entropy loss *without reduction* for causal language models, 
    shifting logits and labels for GPT-like models.
    """
    # Shift logits and labels: logits predict the next token, so remove the last timestep,
    # while labels and mask are shifted left to exclude the first token.
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    mask = mask[..., 1:].contiguous()

    assert logits.size(0) == labels.size(0), "Batch size mismatch between logits and labels."
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)
    loss = cse_loss(logits_flat, labels_flat)
    return loss[mask_flat]

def topk(k: int, num_masked_tokens: int, masked_label_ids: torch.Tensor, masked_prbs: torch.Tensor) -> torch.Tensor:
    """
    Return a float tensor indicating whether the gold label is within the top k 
    predictions for each masked token.
    """
    # Broadcast gold labels to compare with the top-k indices.
    masked_label_ids = masked_label_ids.unsqueeze(-1)
    label_ids_k = torch.broadcast_to(masked_label_ids, (num_masked_tokens, k))
    topk = torch.topk(masked_prbs, k)
    # Check if the gold label appears among the top k indices.
    is_in_topk = (topk.indices == label_ids_k).any(dim=-1)
    acc = is_in_topk.float()
    return acc

def acc_func(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    """
    Compute the number of correct predictions for top-1, top-5, and top-10 for the masked tokens.
    
    Returns:
        top1_acc, top5_acc, top10_acc, num_masked_tokens
    """
    # NOTE: in language modeling evaluation, multiple tokens could be masked in one sequence
    num_masked_tokens =  mask.sum().item()
    prbs = F.softmax(logits, dim=-1)
    masked_prbs = prbs[mask,:]
    masked_label_ids = labels[mask]

    top1_acc = topk(1, num_masked_tokens, masked_label_ids, masked_prbs).sum().item()
    top5_acc = topk(5, num_masked_tokens, masked_label_ids, masked_prbs).sum().item()
    top10_acc = topk(10, num_masked_tokens, masked_label_ids, masked_prbs).sum().item() 

    return top1_acc, top5_acc, top10_acc, num_masked_tokens

def rank_prob_func(logits: torch.Tensor, label_ids: torch.Tensor, mask: torch.Tensor):
    """
    Compute the rank (i.e., position) of the gold label and the log-probability of the gold label for each masked token.
    
    Returns:
        gold_label_ranks: A tensor of ranks (lower is better).
        gold_label_prbs: A tensor of the log-probabilities for the gold labels.
    """
    # Get masked pred probs and gold label ids
    prbs = F.softmax(logits, dim=-1)
    masked_prbs = prbs[mask,:]
    masked_label_ids = label_ids[mask]

    # Get ranks by sorting
    sorted_indices = torch.argsort(masked_prbs, descending=True)
    gold_label_ranks = torch.where(sorted_indices == masked_label_ids.view(-1, 1))[1]
    gold_label_ranks = gold_label_ranks.float()

    # Get probs through corresponding gold label id
    gold_label_prbs = torch.gather(masked_prbs, -1, masked_label_ids.view(-1, 1))
    gold_label_prbs = torch.log(gold_label_prbs.squeeze())    
    
    return gold_label_ranks, gold_label_prbs

# TODO: Delete this if you don't find instances of it being used.
# def qual_func(input_str, logits, label_ids, mask, tokenizer):
#     qual_res = []

#     # getting pred probs
#     prbs = F.softmax(logits, dim=-1)
#     masked_prbs = prbs[mask,:]
#     masked_label_ids = label_ids[mask]

#     # decoding ids into strings
#     pred_ids = torch.argmax(masked_prbs, dim=1)
#     pred_tokens = tokenizer.batch_decode(pred_ids)
#     label_tokens = tokenizer.batch_decode(masked_label_ids)

#     # calc metrics
#     batch_perp = torch.exp(batch_loss_func(logits, label_ids, mask, tokenizer.vocab_size))

#     # NOTE: just top 10 choices
#     masked_label_ids = masked_label_ids.unsqueeze(-1)
#     top10ids = torch.topk(masked_prbs, 10).indices
#     top100ids = torch.topk(masked_prbs, 100).indices

#     # accumulate metrics
#     for i in range(7):
#         qual_res.append(
#             {
#                 "inp_sentence": input_str[i],
#                 "label_token": label_tokens[i],
#                 "pred_token": pred_tokens[i],
#                 "perp": batch_perp[i].item(),
#                 "top10options": " ".join(tokenizer.convert_ids_to_tokens(top10ids[i])),
#                 "top100options": " ".join(tokenizer.convert_ids_to_tokens(top100ids[i])),
#             }
#         )
#     return qual_res


# TODO: Delete this if you don't find instances of it being used.
# def mask_distribution_func(input_str, logits, label_ids, mask, tokenizer, batch_size, k=100):
#     mask_dist_list = []

#     # getting pred probs
#     prbs = F.softmax(logits, dim=-1)
#     masked_prbs = prbs[mask,:]

#     # gold label ids & strs
#     masked_label_ids = label_ids[mask]
#     gold_label_tokens = tokenizer.batch_decode(masked_label_ids)

#     # pred ids & str
#     # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
#     top100_pred_ids = torch.topk(masked_prbs, k=k, dim=-1)
#     for i in range(batch_size):

#         mask_dist_list.append(
#             {
#                 "inp_sentence": input_str[i],
#                 # (token_id, token_str, probability)
#                 # "top_100_id": [top100_pred_ids.indices[i][j].item() for j in range(k)],
#                 "top_100_token": [tokenizer.decode(top100_pred_ids.indices[i][j]) for j in range(k)], 
#                 "top100prbs": [top100_pred_ids.values[i][j].item() for j in range(k)], 
#                 #
#                 "gold_label_id": masked_label_ids[i].item(),
#                 "gold_label_token": gold_label_tokens[i],
#                 #
#                 # "distribution": masked_prbs[i].tolist()
#             }
#         )
#     return mask_dist_list

@torch.no_grad()
def sparsity_func(model: Union[SelectivePrunedQwenLM], lm_name: str, non_mask_counts: bool = False) -> Dict[str, Any]:
    """
    Compute various sparsity statistics for the given model.
    
    Args:
        model: The model instance that provides sparsity methods and attributes.
        lm_name: The language model name (e.g., "gpt2"). Only GPT-based models are supported.
        non_mask_counts: If True, include non-masked counts in the sparsity computations (e.g. if masking only half of the model, may want to instead get sparsity over all the model, not just the masked half).
    
    Returns:
        A dictionary containing sparsity statistics and per-layer/per-head details.
    """
    layers = model.num_layers
    heads = model.num_heads
    
    model.eval()
    layer_sparsity_stats = []
    att_head_sparsity_stats = []

    for layer in range(layers):
        layer_dict = {}

        # 1) overall attention sparsity
        att_sparsity = model.get_mask_sparsity_attn(layer, non_mask_counts)
        layer_dict.update({'layer': layer + 1, 'att_sparsity': att_sparsity})

        # 2) dense layer sparsities (level-specific)
        att_out_or_interm, intermediate, dense_out = model.get_mask_sparsity_mlp(layer, non_mask_counts)
        layer_dict.update({
            'dense1': att_out_or_interm,
            'dense2': intermediate,
            'dense3': dense_out,
        })
        # 3) complete transformer layer sparsity
        layer_sparsity = model.get_mask_sparsity_layer(layer, non_mask_counts)
        layer_dict.update({'layer_sparsity': layer_sparsity})

        layer_sparsity_stats.append(layer_dict)

        # 4) q k v specific + overall attention head sparsities
        for head in range(heads):
            q, k, v = None, None, None
            att_head_sparsity_stats.append({
                'layer': layer + 1,
                'head': head + 1,
                'q': q,
                'k': k,
                'v': v,
            })

    valid_att_list = [layer_dict['att_sparsity'] for layer_dict in layer_sparsity_stats if layer_dict['att_sparsity'] is not None]
    avg_attention = sum(valid_att_list) / len(valid_att_list)
    
    if lm_name.startswith("gpt"):
        avg_dense1 = None
    else:
        valid_dense1_list = [layer_dict['dense1'] for layer_dict in layer_sparsity_stats if layer_dict['dense1'] is not None]
        avg_dense1 = sum(valid_dense1_list) / len(valid_dense1_list)
    
    valid_dense2_list = [layer_dict['dense2'] for layer_dict in layer_sparsity_stats if layer_dict['dense2'] is not None]
    avg_dense2 = sum(valid_dense2_list) / len(valid_dense2_list)
    
    valid_dense3_list = [layer_dict['dense3'] for layer_dict in layer_sparsity_stats if layer_dict['dense3'] is not None]
    avg_dense3 = sum(valid_dense3_list) / len(valid_dense3_list)
    
    valid_layer_list = [layer_dict['layer_sparsity'] for layer_dict in layer_sparsity_stats if layer_dict['layer_sparsity'] is not None]
    avg_layer = sum(valid_layer_list) / len(valid_layer_list)
    
    avg_q = None
    avg_k = None
    avg_v = None

    produced_mask_binary = model.compute_binary_pct_produced_mask_stats()
    
    sparsity_stats_dict = {
        'pct_binary_strict': produced_mask_binary['strict-binary'],
        'pct_binary_non_strict': produced_mask_binary['non-strict-binary'],
        'pct_binary_produced_mask_0': produced_mask_binary["0"],
        'pct_binary_produced_mask_1': produced_mask_binary["1"],
        'pct_binary_produced_mask_?': produced_mask_binary["?"],
        'pct_binary_produced_mask_01': produced_mask_binary["<=0.01"],
        'pct_binary_produced_mask_99': produced_mask_binary[">=0.99"],
        'pct_binary_produced_mask_20': produced_mask_binary["<=0.20"],
        'pct_binary_produced_mask_80': produced_mask_binary[">=0.80"],
        'pct_binary_produced_mask_45_55': produced_mask_binary["0.45-0.55"],
        #
        'unconditonal_sparsity': produced_mask_binary["unconditonal_sparsity"],
        #
        'neuron-sparsity': produced_mask_binary["neuron-sparsity"],
        #
        'sparsity_avg_attention': avg_attention,
        'sparsity_avg_dense1': avg_dense1,
        'sparsity_avg_dense2': avg_dense2,
        'sparsity_avg_dense3': avg_dense3,
        'sparsity_avg_layer': avg_layer,
        'sparsity_avg_q': avg_q,
        'sparsity_avg_k': avg_k,
        'sparsity_avg_v': avg_v,
        #
        'layer_sparsity_stats': layer_sparsity_stats,
        'att_head_sparsity_stats': att_head_sparsity_stats,
    }

    return sparsity_stats_dict
