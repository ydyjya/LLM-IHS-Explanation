# Intermediate Hidden States Explanation (IHS-Explanation)

This repository is the implementation of the paper, [How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States](http://arxiv.org/abs/2406.05644).
With this repository, you can swiftly deploy analysis for LLMs through intermediate hidden states and visualization results.

## Abstract

## Overview

### Supported Models and Technique

|           LLMs            | Weak-to-Strong Explanation |
|:-------------------------:|:--------------------------:|
|    Llama-2-7b-chat-hf     |             ✅              |
|    Llama-2-13b-chat-hf    |             ✅              |
|    Llama-2-70b-chat-hf    |             ✅              |
|      vicuna-7b-v1.5       |             ✅              |
|      vicuna-13b-v1.5      |             ✅              |
| Meta-Llama-3-8B-Instruct  |             ✅              |
| Meta-Llama-3-70B-Instruct |             ✅              |
|       Llama-2-7b-hf       |             ✅              |
|      Llama-2-13b-hf       |             ✅              |
|      Llama-2-70b-hf       |             ✅              |
|      Meta-Llama-3-8B      |             ✅              |
|     Meta-Llama-3-70B      |             ✅              |
| Mistral-7B-Instruct-v0.1  |             ✅              |
| Mistral-7B-Instruct-v0.2  |             ✅              |
|      Mistral-7B-v0.1      |             ✅              |
|    falcon-7b-instruct     |             ✅              |
|         falcon-7b         |             ✅              |

### Supported Weak Classifiers For Weak2Strong Explanation
| Weak Classifier |
|:---------------:|
|       SVM       |
|       MLP       |

### Recommend Implementation For Logit Grafting

When we conducted experiment, we directly modified the source code (for example [modeling_llama.py](https://github.com/huggingface/transformers/blob/2b9e252b16396c926dad0e3c31802b4af8004e93/src/transformers/models/llama/modeling_llama.py)).
We try to inherit a model class and rewrite the forward method for Logit Grafting directly, but there are always some tough bugs that we can't solve. 
So, if you want to reproduce the Logit Grafting, 
we recommend you modified the source code, we will show our modification in ```./resource/modeling_llama.py```. 
This may be inconvenient, and we are very sorry️ for that☹️.
The main modifications are add parameters
```       
logit_grafting = False,
graft_hidden_states = None,
layer2graft = None,
``` 
in the function `forward` in `class LlamaModel`
then, we just modify
```
if idx == layer2graft and logit_grafting:
    hidden_states[:,-1,:] = graft_hidden_states
else:
    hidden_states = layer_outputs[0]
```
if you try to reproduce the Logit Grafting, we suggest you just graft the first token to get a positive token.
(Unless you have similar distributions across multiple tokens.)

---
