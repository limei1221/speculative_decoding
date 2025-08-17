import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


def decoding(model, tokenizer, prompt, draft_model=None, gamma=5, do_sample=True, temperature=0.6, max_new_tokens=128):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = model_input.input_ids
    attention_mask = model_input.attention_mask

    input_len = input_ids.shape[1]

    if draft_model is None:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        return completion
    else:
        draft_model = draft_model.to(device)
        draft_model.eval()

        total_proposed = 0
        total_accepted = 0
        iteration_count = 0
        while iteration_count < 100:
            iteration_count += 1

            with torch.no_grad():
                draft_outputs = draft_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gamma,
                    do_sample=do_sample,
                    temperature=temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            draft_tokens = draft_outputs.sequences[:, input_ids.size(1):]  # torch.Size([1, 5])
            draft_probs = torch.stack(draft_outputs.scores).softmax(-1).squeeze(1)  # torch.Size([5, 151936])

            with torch.no_grad():
                target_outputs = model(
                    torch.cat([input_ids, draft_tokens], dim=1),
                    attention_mask=torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1),
                    return_dict=True,
                )
            target_logits = target_outputs.logits[:, input_ids.size(1) - 1:]
            target_probs = torch.softmax(target_logits, dim=-1).squeeze(0)  # torch.Size([6, 151936])

            accepted_token = []
            for i in range(len(draft_tokens[0])):
                draft_token = draft_tokens[0][i].item()
                target_token = torch.argmax(target_probs[i]).item()
                if draft_token == target_token:
                    accepted_token.append(draft_token)
                else:
                    break
            
            num_accepted = len(accepted_token)
            total_proposed += len(draft_tokens[0])
            total_accepted += num_accepted

            next_token = torch.argmax(target_probs[num_accepted]).item()
            accepted_token.append(next_token)

            if tokenizer.eos_token_id in accepted_token:
                break
            
            accepted_token_tensor = torch.tensor([accepted_token], device=input_ids.device)
            input_ids = torch.cat([input_ids, accepted_token_tensor], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(accepted_token_tensor)], dim=1)

            if input_ids.shape[1] - input_len > max_new_tokens:
                break

        completion = tokenizer.decode(input_ids[0][input_len:], skip_special_tokens=True) 
        return completion, total_proposed, total_accepted


if __name__ == "__main__":
    print("Loading models...")
    model_name = "Qwen/Qwen3-1.7B"
    draft_model_name = "Qwen/Qwen3-0.6B"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    assert tokenizer.vocab_size == draft_tokenizer.vocab_size
    print(f"Models loaded successfully. Vocabulary size: {tokenizer.vocab_size}")

    prompt = "introduce yourself"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
        )
    print(f"Prompt: {prompt}")

    print("\nRunning greedy decoding...")
    start_time = time.time()
    greedy_output = decoding(model, tokenizer, prompt, do_sample=False, temperature=0.6, max_new_tokens=128)
    end_time = time.time()
    print('*** Sample Greedy Decoding Output: ', greedy_output)
    print(f'*** Greedy Decoding Time: {end_time - start_time:.2f} seconds')

    print("\nRunning speculative decoding...")
    start_time = time.time()
    spec_output, total_proposed, total_accepted = decoding(model, tokenizer, prompt, draft_model=draft_model, do_sample=True, temperature=0.6, max_new_tokens=128)
    end_time = time.time()
    print('*** Sample Speculative Decoding Output: ', spec_output)
    print(f'*** Speculative Decoding Time: {end_time - start_time:.2f} seconds')
    print('*** Total Proposed: ', total_proposed)
    print('*** Total Accepted: ', total_accepted)
    print(f'*** Total Accepted Ratio: {total_accepted / total_proposed:.2f}')
