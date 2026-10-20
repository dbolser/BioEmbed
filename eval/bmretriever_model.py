"""BMRetriever (Xu et al., EMNLP 2024) for mteb 2.11, encoded as in the
authors' eval.py (github.com/ritaranx/BMRetriever): query
'{instruction}\\nQuery: {text}', passage 'Represent this passage\\npassage: {text}',
truncate to 511 tokens, append EOS, last-token pooling, dot product on the
unnormalised vectors. Two differences from eval.py: it computes under fp16
autocast on GPU, here it is fp32 on CPU and bf16 on GPU; and passages are our
task text 'title\\n\\nabstract', not its (title + ' ' + text).strip().

The instruction is mteb's task instruction (the same one Qwen3-Embedding and
F2LLM get); symmetric tasks use the query form, as mteb's own BMRetriever
template does.

mteb 2.11.6 registers these models, but its wrapper does not build under
sentence-transformers 6 (Transformer(device=...)), and it would truncate at
2048-32768 tokens and score by cosine. The model card's snippet appends EOS to
rows of a 'pt' tensor, which fails (Tensor + list); eval.py appends it to the
token lists, and that is what is done here.

Run: get_model("BMRetriever/BMRetriever-410M") in run_embedders.py.
"""

import mteb
import numpy as np
import torch
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.bmretriever_models import instruction_template
from mteb.models.model_meta import ScoringFunction
from mteb.types import PromptType
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 512


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Verbatim from the authors' eval.py."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), sequence_lengths]


class BMRetriever(AbsEncoder):
    instruction_template = staticmethod(instruction_template)

    def __init__(self, name: str) -> None:
        meta = mteb.get_model_meta(name)
        self.mteb_model_meta = meta.model_copy(update=dict(  # mteb's loader_kwargs are unused
            similarity_fn_name=ScoringFunction.DOT_PRODUCT, max_tokens=MAX_LEN, loader_kwargs={},
            loader=lambda *a, **k: self))
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=meta.revision)
        if self.tokenizer.pad_token is None:  # as eval.py
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        gpu = torch.cuda.is_available()  # bf16 on GPU, as haystack.py casts it anyway (7B fp32 = 28 GB)
        self.model = AutoModel.from_pretrained(name, revision=meta.revision,
                                               dtype=torch.bfloat16 if gpu else torch.float32)
        self.model.to("cuda" if gpu else "cpu").eval()

    @torch.inference_mode()
    def embed(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """eval.py's SentenceBERT.encode, batched longest-first to cut padding."""
        tok = self.tokenizer
        ids = tok(texts, max_length=MAX_LEN - 1, truncation=True, return_attention_mask=False,
                  return_token_type_ids=False)["input_ids"]
        ids = [x + [tok.eos_token_id] for x in ids]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = np.empty((len(ids), self.model.config.hidden_size), dtype=np.float32)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True,
                            return_attention_mask=True, return_tensors="pt").to(self.model.device)
            h = self.model(**batch).last_hidden_state
            out[idx] = last_token_pool(h, batch["attention_mask"]).float().cpu().numpy()
        return out

    def encode(self, inputs, *, task_metadata, hf_split=None, hf_subset=None,
               prompt_type: PromptType | None = None, **kwargs) -> np.ndarray:
        prefix = self.get_task_instruction(task_metadata, prompt_type)
        return self.embed([prefix + t for batch in inputs for t in batch["text"]],
                          batch_size=kwargs.get("batch_size", 8))
