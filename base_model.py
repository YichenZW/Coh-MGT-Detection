import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)


class BaseModel(PreTrainedModel):
    def __init__(self, args, task2num):
        self.args = args
        num_label = task2num[args.dataset.loader_path]
        config = AutoConfig.from_pretrained(args.model.name, num_labels=num_label)
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model.name, use_fast=False, num_labels=num_label
        )
        self.pretrain_model = AutoModelForSequenceClassification.from_pretrained(
            args.model.name, num_labels=num_label
        )
        self.config = self.pretrain_model.config

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs
