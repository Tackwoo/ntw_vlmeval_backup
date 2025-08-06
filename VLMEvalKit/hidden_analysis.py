#Efficient Streaming Language Models with Attention Sinks

import os
import json
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import gc
import string
import copy as cp
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch.nn.functional as F


def top_k_layerwise(lm_head, processor, sample, top_k=5, save_idx=0):

    os.makedirs(f"topk_vis_hidden/{save_idx}", exist_ok=True)
    os.makedirs(f"topk_vis_token/{save_idx}", exist_ok=True)
    os.makedirs(f"topk_vis_token_masked/{save_idx}", exist_ok=True)

    num_layers = len(sample[1])       # 각 step의 layer 수
    num_steps = len(sample) - 1       # 첫 step 제외

    top1_probs = torch.zeros((num_layers, num_steps))  # [layer, step]
    top1_tokens = [["" for _ in range(num_steps)] for _ in range(num_layers)]
    global_max_hidden = float(70)

    for step_idx, step in enumerate(sample[1:], start=1):
        hidden_fig, hidden_ax = plt.subplots(figsize=(16, 8))
        token_fig, token_ax = plt.subplots(figsize=(16, 8))
        token_fig_masked, token_ax_masked = plt.subplots(figsize=(16, 8))

        for layer_idx, layer in enumerate(step):
            layer_clean = layer.squeeze(0).squeeze(0)
            topk_hidden = torch.topk(layer_clean, k=top_k)
            h_indices = topk_hidden.indices.tolist()
            h_values = topk_hidden.values.tolist()
            max_val = topk_hidden.values.max().item()
            global_max_hidden = max(global_max_hidden, max_val)

            for rank, (idx, val) in enumerate(zip(h_indices, h_values), 1):
                color = cm.Blues(val / global_max_hidden * 0.7 + 0.3)
                hidden_ax.text(layer_idx, rank,
                               f"{idx}\n{val:.2f}",
                               ha='center', va='center',
                               fontsize=9,
                               bbox=dict(facecolor=color, edgecolor='gray', boxstyle='round,pad=0.3'))

            logits = lm_head(layer)
            probs = F.softmax(logits.view(-1), dim=-1)
            topk_token = torch.topk(probs, k=top_k)
            t_indices = topk_token.indices.tolist()
            t_values = topk_token.values.tolist()

            for rank, (idx, val) in enumerate(zip(t_indices, t_values), 1):
                token_str = processor.decode([idx])
                color = cm.Reds(val * 0.7 + 0.3)
                token_ax.text(layer_idx, rank,
                              f"{token_str}\n{val:.2f}",
                              ha='center', va='center',
                              fontsize=9,
                              bbox=dict(facecolor=color, edgecolor='gray', boxstyle='round,pad=0.3'))

            top1_value, top1_index = torch.topk(probs, k=1)
            top1_token = processor.decode([top1_index.item()])
            top1_probs[layer_idx][step_idx - 1] = top1_value.item()
            top1_tokens[layer_idx][step_idx - 1] = top1_token

        for ax, title in [(hidden_ax, "Top-k Hidden Activations"), (token_ax, "Top-k Token Probabilities")]:
            ax.set_xlim(-1, len(step))
            ax.set_ylim(0.5, top_k + 0.7)
            ax.set_yticks(range(1, top_k + 1))
            ax.set_yticklabels([f"Top-{i}" for i in range(1, top_k + 1)])
            ax.set_xlabel("Layer")
            ax.set_title(f"{title} (Step {step_idx})")
            ax.invert_yaxis()
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)

        hidden_fig.tight_layout()
        hidden_fig.savefig(f"topk_vis_hidden/{save_idx}/topk_hidden_step{step_idx}.png")
        plt.close(hidden_fig)

        token_fig.tight_layout()
        token_fig.savefig(f"topk_vis_token/{save_idx}/topk_token_step{step_idx}.png")
        plt.close(token_fig)

        annot_text = [
            [f"{token}\n({prob:.2f})" for token, prob in zip(token_row, prob_row)]
            for token_row, prob_row in zip(top1_tokens, top1_probs.tolist())
        ]

    fig_width = max(20, num_steps * 1)
    fig_height = max(8, num_layers * 0.5) 

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        top1_probs.numpy(),
        annot=annot_text,
        fmt='',
        cmap='YlGnBu',
        xticklabels=[f"Step {i+1}" for i in range(num_steps)],
        yticklabels=[f"Layer {i}" for i in range(num_layers)],
        cbar_kws={"label": "Top-1 Token Probability"}
    )

    plt.title("Top-1 Token + Probability Heatmap", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Layer", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"topk_vis_token/{save_idx}/top1_token_heatmap.png", dpi=300)
    plt.close()
            

