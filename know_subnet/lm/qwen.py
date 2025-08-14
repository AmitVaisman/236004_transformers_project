import torch.nn as nn
import tqdm as tqdm
from typing import List, Type
import torch

from know_subnet.lm.lm_mixins import (
    MaskOpsMixin,
    SelectivePruningMixin,
    MaskStatsMixin
)


from transformers import (
    Qwen2ForCausalLM,
    Qwen2Tokenizer
)

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2DecoderLayer,
)


from transformers.models.qwen2.configuration_qwen2 import (
    Qwen2Config
)


class QwenLM(nn.Module, MaskOpsMixin):
    def __init__(self, use_dropout: bool = False, lm_name: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', dtype=torch.float16):
        """
        Args:
            use_dropout (bool, optional): Whether to use dropout in the model. Defaults to False.
            lm_name (str, optional): Pretrained model name to load.. Defaults to 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'.
        """
        super().__init__()
        self.config = Qwen2Config.from_pretrained(lm_name)
        if not use_dropout:
            self.config.resid_pdrop=0.0
            self.config.embd_pdrop=0.0
            self.config.attn_pdrop=0.0
            self.config.summary_first_dropout=0.0

        self.lm = Qwen2ForCausalLM.from_pretrained(lm_name, config=self.config, torch_dtype=dtype)

        self.lm_name = lm_name
        self.num_layers = self.lm.config.num_hidden_layers
        self.num_heads = self.lm.config.num_attention_heads

    def forward(self, input_ids=None, attention_mask=None,  labels=None):
        return self.lm(input_ids, attention_mask=attention_mask, labels=labels)

    def freeze_params(
        self,
        is_freeze: bool = True,
        exclude_name_list: List[str] = ["mask_scores", "multiple_choice_head"],
        verbose: bool = True
    ):
        """
        Freeze or unfreeze model parameters except for those specified in the exclusion list.

        Args:
            is_freeze (bool): If True, freeze parameters; if False, unfreeze them.
            exclude_name_list (List[str]): List of parameter name substrings to exclude from freezing.
            verbose (bool): If True, print the names of parameters as they are frozen/unfrozen.
        """
        for name, param in self.named_parameters():
            if any(exclude_name in name for exclude_name in exclude_name_list):
                continue  # Skip freezing this parameter
            param.requires_grad = not is_freeze
            if verbose:
                if is_freeze:
                    print("Froze {}".format(name))
                else:
                    print("Unfroze {}".format(name))


class SelectivePrunedQwenLM(QwenLM, SelectivePruningMixin, MaskStatsMixin):
    def __init__(
        self,
        out_w_per_mask: int,
        in_w_per_mask: int,
        lm_name: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        top_k_layers: int = 12,
        linear_types_to_mask: List[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        module_types_to_mask: List[Type] = [Qwen2Attention, Qwen2MLP, Qwen2DecoderLayer],
        use_dropout: bool = False,
        initial_mask_p: float = 0.88,
        top_limit: int = -1,
        bottom_limit: int = -1,
        verbose: bool = False,
    ):
        """
        Args:
            out_w_per_mask (int): Number of output features to mask per parameter.
            in_w_per_mask (int): Number of input features to mask per parameter.
            lm_name (str): Pretrained model name to load. Defaults to 'gpt2'.
            top_k_layers (int): Number of top layers. Defaults to 12.
            linear_types_to_mask (List[str]): List of linear layer types to mask. Defaults to ['c_attn', 'q_attn', 'c_proj', 'c_fc'].
            module_types_to_mask (List[Type]): List of module types to apply masks to. Defaults to [GPT2Attention, GPT2MLP, GPT2Block].
            use_dropout (bool): Whether to use dropout in the model. Defaults to False.
            initial_mask_p (float): Initial probability for applying a mask. Defaults to 0.88.
            top_limit (int): Upper limit for mask pruning (-1 indicates no limit). Defaults to -1.
            bottom_limit (int): Lower limit for mask pruning (-1 indicates no limit). Defaults to -1.
            verbose (bool): If True, enable verbose logging during layer replacement. Defaults to False.
        """
        super().__init__(use_dropout, lm_name)
        
        # Validate input parameters
        self._validate_params(top_k_layers, linear_types_to_mask, module_types_to_mask, top_limit, bottom_limit)
        
        # Replace layers with masked versions
        self.replace_layers_with_masked(
            out_w_per_mask, 
            in_w_per_mask, 
            top_k_layers, 
            linear_types_to_mask, 
            module_types_to_mask, 
            initial_mask_p,
            top_limit,
            bottom_limit,
            verbose
        )
