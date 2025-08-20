import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from typing import Tuple, Optional, Dict, Any, List


def truncate_kv(cache, keep_len):
    assert isinstance(cache, DynamicCache)
    for idx in range(len(cache.layers)):
        cache.layers[idx].keys = cache.layers[idx].keys[..., :keep_len, :]
        cache.layers[idx].values = cache.layers[idx].values[..., :keep_len, :]

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    gamma: int = 5
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 128
    early_stop: bool = True
    verbose: bool = False


class SpeculativeDecoder:
    """Speculative decoding implementation."""

    def __init__(self, target_model, draft_model, tokenizer, config: SpeculativeConfig, device: str = "cuda"):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.target_kv_cache = None
        self.draft_kv_cache = None
        self.next_token = None

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities using the same temperature and filtering.

        Accepts logits of shape [..., vocab] and returns probs with the same leading shape.
        """
        if self.config.temperature > 0.0:
            scaled = logits / self.config.temperature
            if self.config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(scaled, self.config.top_k, dim=-1)
                filtered = torch.full_like(scaled, float("-inf"))
                filtered.scatter_(-1, top_k_indices, top_k_logits)
                scaled = filtered
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                scaled = scaled.masked_fill(indices_to_remove, float("-inf"))
            probs = F.softmax(scaled, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs

    def _sample_token_and_prob(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a token and return its probability from logits shaped [1, vocab]."""
        probs = self._logits_to_probs(logits)
        if self.config.temperature > 0.0:
            tok = torch.multinomial(probs, num_samples=1)
        else:
            tok = torch.argmax(probs, dim=-1, keepdim=True)  # [1, 1]
        return tok, probs

    def _generate_draft_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens using incremental cached decoding on the draft model.

        Returns:
            draft_tokens: [1, gamma]
            draft_token_probs: [1, gamma, vocab_size]
        """
        tokens: List[int] = []
        token_probs: List[torch.Tensor] = []
        tok = self.next_token
        with torch.inference_mode():
            for _ in range(self.config.gamma):
                outputs = self.draft_model(
                    tok,
                    use_cache=True,
                    past_key_values=self.draft_kv_cache,
                    return_dict=False,
                )  # self.draft_kv_cache updated automatically
                logits = outputs[0][:, -1, :]
                tok, tok_prob = self._sample_token_and_prob(logits)
                tokens.append(tok.item())
                token_probs.append(tok_prob)
        draft_tokens = torch.tensor([tokens], device=self.device, dtype=torch.long)
        draft_token_probs = torch.stack(token_probs, dim=0).transpose(0, 1)
        return draft_tokens, draft_token_probs

    def _verify_draft_tokens(
        self, draft_tokens: torch.Tensor, draft_token_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Verify draft tokens with a single target forward pass."""
        accepted_ids: List[int] = []
        num_accepted = 0
        with torch.inference_mode():
            seq = torch.cat([self.next_token, draft_tokens], dim=1)
            outputs = self.target_model(
                seq,
                use_cache=True,
                past_key_values=self.target_kv_cache,
                return_dict=False,
            )  # self.target_kv_cache updated automatically
            target_token_logits = outputs[0]  # [1, gamma+1, vocab_size]
            target_token_probs = self._logits_to_probs(target_token_logits)  # [1, gamma+1, vocab_size]

            for i in range(self.config.gamma):
                draft_tok = draft_tokens[:, i : i + 1]  # [1, 1]
                cur_draft_probs = draft_token_probs[:, i, :]  # [1, vocab_size]
                draft_prob = cur_draft_probs.gather(-1, draft_tok).squeeze()

                cur_target_probs = target_token_probs[:, i, :]  # [1, vocab_size]
                target_prob = cur_target_probs.gather(-1, draft_tok).squeeze()
                if self.config.temperature == 0.0:
                    target_tok = torch.argmax(cur_target_probs, dim=-1)
                    if target_tok.item() == draft_tok.item():
                        accepted_ids.append(draft_tok.item())
                        num_accepted += 1
                        continue
                else:
                    acceptance_prob = min((target_prob / draft_prob).item(), 1.0)
                    if torch.rand(1, device=self.device).item() < acceptance_prob:
                        accepted_ids.append(draft_tok.item())
                        num_accepted += 1
                        continue
                break

            if num_accepted < self.config.gamma:
                adjusted_probs = torch.clamp(
                    target_token_probs[:, num_accepted, :] - draft_token_probs[:, num_accepted, :],
                    min=0,
                )
                next_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)
            else:
                next_probs = target_token_probs[:, num_accepted, :]

            if self.config.temperature == 0.0:
                next_token = torch.argmax(next_probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(next_probs, num_samples=1)

        return (
            torch.tensor([accepted_ids], device=self.device, dtype=torch.long),
            num_accepted,
            next_token,
        )

    def decode(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Main decoding function with speculative decoding and optimizations."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_tokens = []
        # prefilling
        with torch.inference_mode():
            drf_out = self.draft_model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=False,
            )
            self.draft_kv_cache = drf_out[1]

        with torch.inference_mode():
            tgt_out = self.target_model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=False,
            )
            self.target_kv_cache = tgt_out[1]
        keep_len = input_ids.size(1)

        next_logits = tgt_out[0][:, -1, :]
        next_token, _ = self._sample_token_and_prob(next_logits)
        self.next_token = next_token
        generated_tokens.append(self.next_token.item())

        # speculative decoding
        total_proposed = 0
        total_accepted = 0
        while keep_len + 1 < self.config.max_new_tokens:
            draft_tokens, draft_token_probs = self._generate_draft_tokens()

            accepted_tokens, num_accepted, next_token = self._verify_draft_tokens(
                draft_tokens, draft_token_probs
            )

            total_proposed += self.config.gamma
            total_accepted += num_accepted

            # update KV caches
            keep_len += 1 + num_accepted  # [self.next_token, accepted_tokens]
            if num_accepted == self.config.gamma:
                # last token is accepted, so we need to update self.draft_kv_cache with the last token
                with torch.inference_mode():
                    outputs = self.draft_model(
                        accepted_tokens[:, -1:],
                        use_cache=True,
                        past_key_values=self.draft_kv_cache,
                        return_dict=False,
                    )
            else:
                truncate_kv(self.draft_kv_cache, keep_len)
            truncate_kv(self.target_kv_cache, keep_len)

            generated_tokens.extend(accepted_tokens[0].tolist())

            self.next_token = next_token
            generated_tokens.append(self.next_token.item())

            if self.tokenizer.eos_token_id in generated_tokens and self.config.early_stop:
                eos_idx = generated_tokens.index(self.tokenizer.eos_token_id)
                generated_tokens = generated_tokens[:eos_idx]
                break

        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        metrics = {
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "acceptance_ratio": (
                total_accepted / total_proposed if total_proposed > 0 else 0
            ),
            "generated_length": len(generated_tokens),
        }

        return completion, metrics


def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "cuda") -> str:
    """Simple greedy decoding for comparison."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.size(1)

    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    num_generated_tokens = len(outputs[0][input_len:])
    return completion, num_generated_tokens


def greedy_decode_with_cache(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "cuda") -> str:
    """Simple greedy decoding for comparison."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    kv_cache = None

    generated_tokens = []
    with torch.inference_mode():
        # prefill
        model_output = model(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            return_dict=False,
        )
        kv_cache = model_output[1]
        next_token = torch.argmax(model_output[0][:, -1, :], dim=-1)

        generated_tokens.append(next_token.item())
        while len(generated_tokens) < max_new_tokens:
            # deocde
            model_output = model(
                next_token.view(1, 1),
                use_cache=True,
                past_key_values=kv_cache,
                return_dict=False,
            )  # kv_cache updated automatically
            next_token = torch.argmax(model_output[0][:, -1, :], dim=-1)
            generated_tokens.append(next_token.item())

            if tokenizer.eos_token_id in generated_tokens:
                break

    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    num_generated_tokens = len(generated_tokens)
    return completion, num_generated_tokens


if __name__ == "__main__":
    print("Loading models...")
    model_name = "Qwen/Qwen3-4B"  # recommend 100x larger than draft model from the paper
    draft_model_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    target_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name).to(device)
    target_model.eval()
    draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

    assert tokenizer.get_vocab() == draft_tokenizer.get_vocab(), "Vocabulary must match"
    print(f"Models loaded successfully. Vocabulary size: {tokenizer.vocab_size}")

    prompt = "introduce yourself"
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(f"Prompt: {formatted_prompt}")

    config = SpeculativeConfig(
        gamma=3,  # number of draft tokens to generate each iteration
        temperature=0.0,  # temperature=0.0: deterministic, 0.0<temperature<1.0: more sharpness, temperature>1.0: more randomness
        top_p=1,
        top_k=0,
        max_new_tokens=128,
        early_stop=True,
        verbose=False,
    )

    decoder = SpeculativeDecoder(target_model, draft_model, tokenizer, config, device=device)

    print("Running greedy decoding...")
    start_time = time.time()
    greedy_output, greedy_num_tokens = greedy_decode(
        target_model, tokenizer, formatted_prompt, max_new_tokens=128, device=device
    )
    greedy_time = time.time() - start_time
    print(f"Greedy output: {greedy_output!r}")
    print(f"Greedy decoding time: {greedy_time:.2f}s")
    print(f"Greedy throughput: {greedy_num_tokens / greedy_time:.2f} tokens/s")

    print("Running greedy decoding with kv cache...")
    start_time = time.time()
    cache_output, cache_num_tokens = greedy_decode_with_cache(
        target_model, tokenizer, formatted_prompt, max_new_tokens=128, device=device
    )
    cache_time = time.time() - start_time
    print(f"Cache output: {cache_output!r}")
    print(f"Cache decoding time: {cache_time:.2f}s")
    print(f"Cache throughput: {cache_num_tokens / cache_time:.2f} tokens/s")

    print(f"\nRunning speculative decoding (gamma={config.gamma})...")
    start_time = time.time()
    spec_output, metrics = decoder.decode(formatted_prompt)
    spec_time = time.time() - start_time
    print(f"Speculative output: {spec_output!r}")
    print(f"Speculative decoding time: {spec_time:.2f}s")
    print(f"Speculative throughput: {metrics['generated_length'] / spec_time:.2f} tokens/s")
    print(f"Speculative Metrics: {metrics}")

    print(f"Speedup over greedy decoding: {greedy_time / spec_time:.2f}x")
    print(f"Speedup over greedy decoding with kv cache: {cache_time / spec_time:.2f}x")
