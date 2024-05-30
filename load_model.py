import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

TOP_LOGITS_K = 5


def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def step_forward(model, tokenizer, prompt, decoding=True):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
        tl_pair = []
        lm_head = model.lm_head
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            norm = model.transformer.ln_f
        else:
            raise ValueError(f"Incorrect Model")
        for i, r in enumerate(outputs.hidden_states):
            layer_logits = []
            layer_output = norm(r)
            logits = lm_head(layer_output)
            next_token_logits = logits[:, -1, :]
            top_logits_k = TOP_LOGITS_K
            top_values, top_indices = torch.topk(next_token_logits, top_logits_k, dim=-1)
            decoded_texts = [tokenizer.decode([idx], skip_special_tokens=False) for idx in top_indices.squeeze().tolist()]
            top_values = top_values.detach().cpu()
            if decoding:
                for value, token in zip(top_values.squeeze().tolist(), decoded_texts):
                    layer_logits.append([token, value])
            else:
                for value, token in zip(top_values.squeeze().tolist(), top_indices.squeeze().tolist()):
                    layer_logits.append([token, value])
            tl_pair.append(layer_logits)
    res_hidden_states = []
    for _ in outputs.hidden_states:
        res_hidden_states.append(_.detach().cpu().numpy())
    return res_hidden_states, tl_pair
