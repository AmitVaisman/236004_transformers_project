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

from know_subnet.lm.gpt2 import GPT2LM

"""
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.0, inplace=False)
          (resid_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

NOTE:   linear_types_to_mask: List[str] = ['c_attn', 'q_attn', 'c_proj', 'c_fc'], 
        module_types_to_mask: List[Type] = [GPT2Attention, GPT2MLP, GPT2Block],
"""

"""
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)

NOTE:   linear_types_to_mask: List[str] = ['c_attn', 'q_attn', 'c_proj', 'c_fc'], TODO: WHICH ONES?
        module_types_to_mask: List[Type] = [Qwen2Attention, Qwen2MLP, Qwen2DecoderLayer],
"""
use_dropout = True
def main():
    lm_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    config = Qwen2Config.from_pretrained(lm_name)
    if not use_dropout:
        config.update({"attn_pdrop":0.0, "embd_pdrop":0.0, "resid_pdrop":0.0})

    lm = Qwen2ForCausalLM.from_pretrained(lm_name, config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained(lm_name)
    num_layers = lm.config.num_hidden_layers
    num_heads = lm.config.num_attention_heads
    print(f"Num layers:{num_layers}, Num Heads:{num_heads}")
    print(lm)
main()