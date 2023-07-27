import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoTokenizer, AutoConfig
from typing import Optional

from text_generation_server.models import CausalLM
from text_generation_server.models.custom_modeling.btlm_modeling import (
    BTLMForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

from loguru import logger

logger = logger.opt(colors=True)

tracer = trace.get_tracer(__name__)


class BTLMSharded(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashBTLM is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames, device=device, dtype=dtype, process_group=self.process_group
        )
        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)


        model = BTLMForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        
    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits
        return logits, outputs.past_key_values
