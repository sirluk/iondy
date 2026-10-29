---
title: "Introducing pytora — a minimal PyTorch implementation of LoRA"
date: "2024-05-26"
description: "A small, flexible implementation of LoRA using native PyTorch parametrizations."
category: "Engineering"
tags: ["fine-tuning", "transformers", "machine-learning", "llm"]
original_url: "https://medium.com/@lukas.hauzenberger/introducing-pytora-a-minimal-pytorch-implementation-of-lora-b4ea527b6ac0"
---

## Introducing pytora — a minimal PyTorch implementation of LoRA



![](<images/5a9cf5ed3482dbdb.png>)



TLDR - here is the code: [https://github.com/sirluk/pytora](<https://github.com/sirluk/pytora>)



Low-Rank Adaptation ([LoRA](<https://arxiv.org/pdf/2106.09685>)) has emerged as a game-changing technique in the realm of large language model fine-tuning. By significantly reducing the number of trainable parameters, LoRA makes adapting massive models to specific tasks or domains more computationally feasible and efficient. Its elegant approach, which injects trainable rank decomposition matrices into existing model layers, has led to widespread adoption across research and industry. The success of LoRA is also evident in the number of variations it inspired: LoRA+, DoRA, VeRA, AdaLoRA, Delta-LoRA, MoRA and many more.



If you are reading this, chances are you have already used LoRA for one or the other task. Probably the most common way to integrate LoRA into ones training code is probably through the 🤗 [PEFT](<https://huggingface.co/docs/peft/en/index>) library. However, you might prefer an alternative approach for various reasons, such as



1. **Limited Flexibility:** PEFT provides a streamlined way to apply LoRA to specific model architectures and tasks. However, it may not offer the same level of customisation or control as a manual implementation, especially for researchers who want to experiment with variations of LoRA or apply it to less common architectures.
2. **Potential Performance Overhead:** In some cases, the abstractions and convenience features of PEFT might introduce a slight performance overhead compared to a highly optimised, bare-bones implementation of LoRA. For applications where maximising computational efficiency is critical, this overhead could be a factor.
3. **Dependency on Hugging Face Ecosystem:** PEFT is deeply integrated with the 🤗 Transformers library and ecosystem. While this is convenient for users who already work within that framework, it could be a drawback for those who prefer other libraries or want to avoid additional dependencies.



I created this guide to provide an easy alternative to PEFT.



## pytora for LoRA



Since Pytorch 1.9 there is a very elegant way to implement LoRA with vanilla PyTorch code: PyTorch parametrizations. They act like forward pre-hooks and allow you to define operations for weights that are executed before each forward pass. For details I recommend the official [tutorial](<https://pytorch.org/tutorials/intermediate/parametrizations.html>) which helped me a lot for the first implementation.



In pytora LoRA adapters are implemented this way. A LoRA layer is defined as a standard torch Module and then applied to each weight matrix which we want to finetune.



```
import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization
import math


class LoraLayer(nn.Module):

    def __init__(self, weight, r, alpha = 1, dropout_prob = 0, fan_in_fan_out = False):
        super().__init__()

        if fan_in_fan_out:
            self.in_features = weight.shape[0]
            self.out_features = weight.shape[1]
        else:
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]
        self.alpha = alpha
        self.fan_in_fan_out = fan_in_fan_out

        if dropout_prob > 0.:
            self.lora_dropout = nn.Dropout(p=dropout_prob)
        else:
            self.lora_dropout = nn.Identity()

        self._init_lora(r, weight_dtype=weight.dtype)

    def _init_lora(self, r, weight_dtype = None):
        # Actual trainable parameters
        if r > 0:
            if weight_dtype == None:
                weight_dtype = self.lora_A.dtype
            self.register_parameter(
                'lora_A',
                nn.Parameter(torch.empty((self.in_features, r), dtype=weight_dtype))
            )
            self.register_parameter(
                'lora_B',
                nn.Parameter(torch.zeros((r, self.out_features), dtype=weight_dtype))
            )
            self.scaling = self.alpha / r
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            try:
                # ensure parameters do not exist if they are zero
                delattr(self, "lora_A")
                delattr(self, "lora_B")
                delattr(self, "scaling")
            except AttributeError:
                pass
        self.r = r

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)

    def forward(self, X):
        if self.r == 0:
            return X
        else:
            lora = self.lora_dropout(self.lora_A @ self.lora_B * self.scaling)
            if not self.fan_in_fan_out:
                lora = lora.T
            return X + lora


# example for applying a lora adapter

module = torch.nn.Linear(10,10)

adapter = LoraLayer(weight = module.weight, r = 4)

register_parametrization(module, "weight", adapter)
```



If you install pytora into your environment you can add LoRA adapters to all or only a subset of the layers in your model with only one line of code.



```
from transformers import AutoModel
from pytora import apply_lora

model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B")

# OPTION 1: apply to all layers
apply_lora(model, lora_r = 4)

# OPTION 2: apply to subset explicit
apply_lora(model, lora_r = 4, names_to_include = ["q_proj", "k_proj"])

# OPTION 3: apply to subset implicit
apply_lora(model, lora_r = 4, names_to_exclude = ["lm_head"])
```



If you just want to use vanilla PyTorch and dont want to install pytora even though it has no dependencies other than “torch&gt;=1.9.0” you can of course also just copy/paste the LoRA module definition and call *register\_parametrizations* in your own code.



## What’s Next for pytora?



This is just the beginning for pytora. I have ambitious plans to expand its capabilities, including:



- **More flexible LoRA adapters:** Support for merged QKV layers and other transformer architectures.
- **Other LoRA variants:** Implementations of proven methods like LoRA+, DoRA, and others.
- **Performance optimizations:** Make pytora even faster and more efficient.



If you’re excited about applying LoRA with vanilla PyTorch, I invite you to join me in improving pytora. Check out the GitHub repository, contribute your ideas and code, or simply share your feedback.
