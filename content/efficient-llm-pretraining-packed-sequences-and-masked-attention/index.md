---
title: "Efficient LLM Pretraining: Packed Sequences and Masked Attention"
date: "2024-10-07"
description: "Making every token count with sequence packing, attention masks, and correctly aligned position IDs."
category: "Research"
tags: []
original_url: "https://medium.com/@lukas.hauzenberger/efficient-llm-pretraining-packed-sequences-and-masked-attention-351d2a43b719"
---

![](<images/e43c055c4b6c384c.png>)

*Credit: [https://huggingface.co/blog/poedator/4d-masks](<https://huggingface.co/blog/poedator/4d-masks>)*



Training large language models (LLMs) is a computationally demanding task. It requires vast amounts of data, powerful hardware, and clever optimization techniques. One such technique which is not often talked about, is the use of *packed sequences* to fully take advantage of the chose context length in each training step.



Imagine feeding a Transformer model a batch of text sequences of varying lengths. To maintain consistent input dimensions, shorter sequences are padded with special tokens. While this seems innocuous, it wastes precious GPU memory by attending to meaningless padding tokens.



## **The solution: sequence packing**



Packed sequences offer an elegant solution. Instead of padding, we concatenate multiple shorter sequences into a single, longer sequence. This minimizes wasted compute (through padding tokens). It also allows us to process more tokens per batch thus reducing training time. However, there’s a catch: we need to ensure the model doesn’t attend across sequence boundaries. Lets have a look at a simple example. We are packing together the following three sentences into a single sequence separated by EOS tokens.



```
# Setup
import torch; torch.set_printoptions(linewidth=200)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config = AutoConfig.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_config(config)
```



```
sentence1 = "The cat sat on the mat"
sentence2 = "The dog ate my homework"
sentence3 = "My aunt is a teacher"

sentences = [sentence1, sentence2, sentence3]
tokenized_sentences = tokenizer(sentences, return_attention_mask=False, add_special_tokens=False)["input_ids"]
tokenized_sentences = [t for s in tokenized_sentences for t in s + [tokenizer.eos_token_id]]
tokenizer.decode(tokenized_sentences)
```



If we would decode the packed sequence again it would look like this:



*The cat sat on the mat&lt;|endoftext|&gt;The dog ate my homework&lt;|endoftext|&gt;My aunt is a teacher&lt;|endoftext|&gt;*



The standard attention mask for causal language modeling for the packed sequences would look like this.



```
tokenized_sentences = torch.tensor(tokenized_sentences)
attn_mask = torch.ones(tokenized_sentences.size(0), tokenized_sentences.size(0), dtype=torch.int).tril()
attn_mask
```



![](<images/d6960b5ce1fccfe5.png>)



With this mask however, when processing the second sentence, the model can still attend to tokens in the first sentence which is not ideal as the two examples are independent. To fix this we can truncate the attention mask in a certain way. When having only one sample in the batch it is relatively easy to do in pytorch.



```
def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
    # store sequence length in variable for easier readability
    T = tokenized_sentences.size(0)
    # get indices of all EOS tokens
    eos_indices = (tokenized_sentences == tokenizer.eos_token_id).nonzero().squeeze()
    # from indices, get length of each sequence
    reps = torch.cat([eos_indices[[0]]+1, eos_indices[1:] - eos_indices[:-1]])
    # repeat each eos index n times along dimension 1 (n is the number of tokens in the sequence)
    repeated_idx = torch.repeat_interleave(eos_indices, reps).view(1,-1).expand(T, -1)
    # create tensor with all indices from 0 to T-1 repeated T times along dimesion 1
    mask_indices = torch.arange(T).view(-1,1).expand(-1, T)
    # create causal mask and additionally mask out all tokens from preceeding sequences
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(-1, -1)
    mask.masked_fill_(mask_indices > repeated_idx, False)
    return mask

get_attention_mask_for_packed_sequence(tokenized_sentences, tokenizer.eos_token_id)
```



![](<images/56378f97b3d06f00.png>)



As you can see, the standard causal mask is truncated to mask out tokens from previous sentences.



## **Adjust position ids accordingly**



When packing sequences together it is important to adjust the position ids use to create position embeddings accordingly. Each token in a sequence typically has an associated position id that helps the model understand the token’s relative position. When we pack multiple sequences together, we need to ensure that the position IDs for each sequence start from the beginning (usually 0 or 1) rather than continuing from where the previous sequence left off.



By adjusting position ids we also clearly mark sequence boundaries. This is crucial for the model to distinguish between different sequences and not treat the packed data as one continuous sequence.



We can leverage the code from above function to generate the tensor with position ids



```
pos_ids = torch.arange(T) - torch.repeat_interleave(torch.cat([torch.tensor([0]), eos_indices+1], dim=0)[:-1], reps)
pos_ids
```



## **Attention mask for batched sequence packing**



Typically during training we would like to process and entire batch of sequences. For the example code above we would have to resort to a loop implementation to get the truncated attention mask for the batch. To do it without a loop is a little bit more challenging due to the additional batch dimension. To show you how to do it lets first create a second item of packed sqeuences to get a batch of size 2.



```
sentence4 = "Rome wasn't built in a day"
sentence5 = "My hovercraft is full of eels"

sentences = [sentence4, sentence5]
tokenized_sentences2 = tokenizer(sentences, return_attention_mask=False, add_special_tokens=False)["input_ids"]
tokenized_sentences2 = torch.tensor([t for s in tokenized_sentences2 for t in s + [tokenizer.eos_token_id]])

batch = torch.nn.utils.rnn.pad_sequence(
  [tokenized_sentences, tokenized_sentences2],
  batch_first=True, padding_value=tokenizer.eos_token_id
)
```



We assign the shape of the batch to two variables B and T. This makes the following code more readable.



```
B, T = batch.shape
```



The main challenge for the batched implementation is to construct the same “repated\_index” tensor like in the example from above. First we need the global indices of the EOS tokens.



```
eos_idx = (batch.view(-1) == tokenizer.eos_token_id) \
  .nonzero(as_tuple=True)[0] + 1
```



![](<images/ffaf09b0936d8704.png>)



To this index vector we add the 0 index and the last token index for each batch item. This is needed to be able to separate the batch items again later on. We then remove duplicates (in case the first or last index for a batch item is already present) and sort.



```
eos_idx_expanded = torch.cat(
  [eos_idx, torch.arange(0,B*T+1,T)]
).unique().sort()[0]
```



![](<images/c8e4d3b615de8a97.png>)



Next since our index vector contains the global indices of EOS tokens within the batch (e.g. the forst index of the second batch item = T) we need to normalize the indices by the sequence length. For the normalized indices we replace zeros with T. This is needed in the following step.



```
normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
```



![](<images/ffaf09b0936d8704.png>)



With the normalized indices we can check how often we need to repeat each EOS token index to get the correct sequence length. To achieve this we needed to have the last index for each sequence present. If we didnt replace 0s with T in the step beforfe the number of repetitions for the last eos index in each batch would be wrong.



```
reps = normalized_idx[1:] - normalized_idx[:-1]
reps = torch.where(reps < 1, normalized_idx[1:], reps)
```



![](<images/c8e4d3b615de8a97.png>)



Now we can create the batched repeated index tensor



```
repeated_idx = torch.repeat_interleave(
  normalized_idx[1:], reps
).view(B,1,T).expand(-1,T,-1)
```



![](<images/55dcaef2f4a329f2.png>)



The rest is similar to the example with batch size = 1. We construct a tensor with indices from 0 to T-1 repeated T times along dimension 1 and create a causal mask. We then mask out all tokens from preceeding sequences.



```
mask_indices = torch.arange(T).view(1,-1,1).expand(B, -1, T)
# create mask
mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1)
mask = mask.masked_fill(mask_indices >= repeated_idx, False)
```



Here is the full function. I added to possiblity to chose between checking eos tokens or bos tokens.



```
def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0,B*T+1,T)]).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    repeated_idx = torch.repeat_interleave(normalized_idx[1:], reps).view(B,1,T).expand(-1,T,-1)
    mask_indices = torch.arange(T).view(1,-1,1).expand(B, -1, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1)
    mask = mask.masked_fill(mask_indices >= repeated_idx, False)
    return mask
```



Analogously to the example above without a batch dimension you can reuse the code for creating the attention mask to get the correct position ids:



```
pos_ids = (torch.arange(B*T) - torch.repeat_interleave(eos_idx_expanded[:-1], reps)).view(B,T)
pos_ids
```



You can also find all code snippets in this [notebook](<https://github.com/sirluk/llm_finetuning/blob/main/attention_mask_for_packed_sequence/tutorial.ipynb>).
