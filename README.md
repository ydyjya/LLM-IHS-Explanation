# Intermediate Hidden States Explanation (IHS-Explanation)

This repository is the implementation of the paper, [How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States](www.arxiv.org).
With this repository, you can swiftly deploy analysis for LLMs through intermediate hidden states and visualization results.


## Overview

### Supported Models and Technique

|           LLMs           | Weak-to-Strong Explanation | Logit Grafting |
|:------------------------:|:--------------------------:|:--------------:|
|    Llama-2-7b-chat-hf    |             ✅              |       ✅        |
|   Llama-2-13b-chat-hf    |             ✅              |       ✅        |
|      vicuna-7b-v1.5      |             ✅              |       ✅        |
|     vicuna-13b-v1.5      |             ✅              |       ✅        |
| Meta-Llama-3-8B-Instruct |             ✅              |       ✅        |
|      Llama-2-7b-hf       |             ✅              |       ✅        |
|      Llama-2-13b-hf      |             ✅              |       ✅        |
|     Meta-Llama-3-8B      |             ✅              |       ✅        |
| Mistral-7B-Instruct-v0.1 |             ✅              |       ❌        |
| Mistral-7B-Instruct-v0.2 |             ✅              |       ❌        |
|     Mistral-7B-v0.1      |             ✅              |       ❌        |
|    falcon-7b-instruct    |             ✅              |       ❌        |
|        falcon-7b         |             ✅              |       ❌        |

### Supported Weak Classifiers For Weak2Strong Explanation
| Weak Classifier |
|:---------------:|
|       SVM       |
|       MLP       |

---
