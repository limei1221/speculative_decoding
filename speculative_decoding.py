import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from typing import Tuple, Optional, Dict, Any, List


def truncate_kv(cache, keep_len):
    truncated = []
    for layer_k, layer_v in cache:
        # works for common HF shapes (batch, num_heads, seq_len, head_dim)
        truncated.append((layer_k[..., :keep_len, :], layer_v[..., :keep_len, :]))
    return DynamicCache.from_legacy_cache(truncated)


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
    max_iterations: int = 100


class SpeculativeDecoder:
    """Speculative decoding implementation."""

    def __init__(self, target_model, draft_model, tokenizer, config: SpeculativeConfig):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.target_model = self.target_model.to(self.device)
        self.draft_model = self.draft_model.to(self.device)

        self.target_model.eval()
        self.draft_model.eval()

        # KV cache state for incremental decoding
        self.target_kv_cache = None
        self.draft_kv_cache = None
        self.next_token = None  # shape [1,1]

    def _sample_token_and_prob(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a token and return its probability from logits shaped [1, vocab]."""
        if self.config.temperature > 0.0:
            scaled = logits / self.config.temperature
            if self.config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    scaled, self.config.top_k, dim=-1
                )
                filtered = torch.full_like(scaled, float("-inf"))
                filtered.scatter_(-1, top_k_indices, top_k_logits)
                scaled = filtered
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                scaled[indices_to_remove] = float("-inf")
            probs = F.softmax(scaled, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
        else:
            probs = F.softmax(logits, dim=-1)
            tok = torch.argmax(probs, dim=-1, keepdim=True)
        tok_prob = probs.gather(-1, tok).squeeze(-1)
        return tok, tok_prob

    def _generate_draft_tokens(self, gamma: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens using incremental cached decoding on the draft model.

        Returns:
            draft_tokens: [1, gamma]
            draft_token_probs: [gamma]
        """
        tokens: List[int] = []
        token_probs: List[torch.Tensor] = []
        local_next = self.next_token
        with torch.inference_mode():
            for _ in range(gamma):
                outputs = self.draft_model(
                    local_next,
                    use_cache=True,
                    past_key_values=self.draft_kv_cache,
                    return_dict=False,
                )
                logits = outputs[0][:, -1, :]
                tok, tok_prob = self._sample_token_and_prob(logits)
                tokens.append(tok.item())
                token_probs.append(tok_prob.squeeze(0))
                self.draft_kv_cache = outputs[1]
                local_next = tok
        draft_tokens = torch.tensor([tokens], device=self.device, dtype=torch.long)
        draft_token_probs = torch.stack(token_probs, dim=0)
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
            )
            logits_per_pos = outputs[0]  # [1, gamma+1, vocab]

            for i in range(draft_tokens.size(1)):
                step_logits = logits_per_pos[:, i, :]
                draft_tok = draft_tokens[:, i : i + 1]
                draft_prob = draft_token_probs[i]
                target_probs = F.softmax(step_logits, dim=-1)
                target_prob = target_probs.gather(-1, draft_tok).squeeze()
                if self.config.temperature == 0.0:
                    target_tok = torch.argmax(target_probs, dim=-1).item()
                    draft_tok = draft_tok.item()
                    if target_tok == draft_tok:
                        accepted_ids.append(draft_tok)
                        num_accepted += 1
                        continue
                else:
                    if target_prob.item() > 0 and draft_prob.item() > 0:
                        acceptance_prob = min((target_prob / draft_prob).item(), 1.0)
                        if torch.rand(1, device=self.device).item() < acceptance_prob:
                            accepted_ids.append(int(draft_tok.item()))
                            num_accepted += 1
                            continue
                break

            next_logits = logits_per_pos[:, num_accepted, :]
            next_token, _ = self._sample_token_and_prob(next_logits)

        return (
            torch.tensor([accepted_ids], device=self.device, dtype=torch.long),
            num_accepted,
            next_token,
        )

    def _adaptive_gamma(self, acceptance_ratio: float) -> int:
        """Adaptively adjust gamma based on acceptance ratio."""
        if acceptance_ratio > 0.8:
            return min(self.config.gamma + 1, 10)
        elif acceptance_ratio < 0.3:
            return max(self.config.gamma - 1, 2)
        else:
            return self.config.gamma

    def decode(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Main decoding function with speculative decoding and optimizations."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_tokens = []

        total_proposed = 0
        total_accepted = 0
        iterations = 0
        current_gamma = self.config.gamma

        # Prefill both models to initialize KV caches
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
            keep_len = self.target_kv_cache.get_seq_length()

        next_token, _ = self._sample_token_and_prob(tgt_out[0][:, -1, :])
        self.next_token = next_token

        while len(generated_tokens) < self.config.max_new_tokens:
            iterations += 1

            if self.config.verbose:
                print(
                    f"Iteration {iterations}: Current length {len(generated_tokens)}, Gamma: {current_gamma}"
                )

            draft_tokens, draft_token_probs = self._generate_draft_tokens(current_gamma)

            accepted_tokens, num_accepted, next_token = self._verify_draft_tokens(
                draft_tokens, draft_token_probs
            )

            total_proposed += draft_tokens.size(1)
            total_accepted += num_accepted

            tokens_to_commit = torch.concat([self.next_token, accepted_tokens], dim=1)

            # Update KV caches properly for DynamicCache objects
            keep_len += 1 + num_accepted
            self.draft_kv_cache = truncate_kv(self.draft_kv_cache, keep_len)
            self.target_kv_cache = truncate_kv(self.target_kv_cache, keep_len)

            self.next_token = next_token

            for token_id in tokens_to_commit[0].tolist():
                generated_tokens.append(token_id)
                if token_id == self.tokenizer.eos_token_id and self.config.early_stop:
                    break

            if (
                self.tokenizer.eos_token_id in generated_tokens
                and self.config.early_stop
            ) or len(generated_tokens) >= self.config.max_new_tokens:
                break

            if iterations > self.config.max_iterations:
                print("Warning: Maximum iterations reached")
                break

            if iterations > 1:
                acceptance_ratio = total_accepted / total_proposed
                current_gamma = self._adaptive_gamma(acceptance_ratio)

        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        metrics = {
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "acceptance_ratio": (
                total_accepted / total_proposed if total_proposed > 0 else 0
            ),
            "iterations": iterations,
            "generated_length": len(generated_tokens),
            "final_gamma": current_gamma,
            "efficiency": len(generated_tokens) / iterations if iterations > 0 else 0,
        }

        return completion, metrics


def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Simple greedy decoding for comparison."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.size(1)
    completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return completion


def greedy_decode_with_cache(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Simple greedy decoding for comparison."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.eval()

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
            )
            kv_cache = model_output[1]
            next_token = torch.argmax(model_output[0][:, -1, :], dim=-1)
            generated_tokens.append(next_token.item())

            if tokenizer.eos_token_id in generated_tokens:
                break

    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return completion


if __name__ == "__main__":
    print("Loading models...")
    model_name = "Qwen/Qwen3-1.7B"
    draft_model_name = "Qwen/Qwen3-0.6B"

    def load_model(name: str):
        if torch.cuda.is_available():
            try:
                return AutoModelForCausalLM.from_pretrained(
                    name, attn_implementation="flash_attention_2"
                )
            except Exception:
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        name, attn_implementation="sdpa"
                    )
                except Exception:
                    return AutoModelForCausalLM.from_pretrained(name)
        else:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    name, attn_implementation="sdpa"
                )
            except Exception:
                return AutoModelForCausalLM.from_pretrained(name)

    target_model = load_model(model_name)
    draft_model = load_model(draft_model_name)

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

    init_gamma = 5
    config = SpeculativeConfig(
        gamma=init_gamma,
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        max_new_tokens=128,
        early_stop=True,
        verbose=False,
    )

    # Optionally compile for speed (CUDA recommended)
    def maybe_compile(model):
        try:
            return torch.compile(model, mode="max-autotune")
        except Exception:
            return model

    if torch.cuda.is_available():
        target_model = maybe_compile(target_model)
        draft_model = maybe_compile(draft_model)

    decoder = SpeculativeDecoder(target_model, draft_model, tokenizer, config)

    print("Running greedy decoding...")
    start_time = time.time()
    greedy_output = greedy_decode(
        target_model, tokenizer, formatted_prompt, max_new_tokens=128
    )
    greedy_time = time.time() - start_time
    print(f"Greedy output: {greedy_output}")
    print(f"Greedy decoding time: {greedy_time:.2f}s")

    print("Running greedy decoding with kv cache...")
    start_time = time.time()
    cache_output = greedy_decode_with_cache(
        target_model, tokenizer, formatted_prompt, max_new_tokens=128
    )
    cache_time = time.time() - start_time
    print(f"Cache output: {cache_output}")
    print(f"Cache decoding time: {cache_time:.2f}s")

    print(f"\nRunning speculative decoding (gamma={init_gamma})...")
    start_time = time.time()
    spec_output, metrics = decoder.decode(formatted_prompt)
    spec_time = time.time() - start_time
    print(f"Speculative output: {spec_output}")
    print(f"Speculative decoding time: {spec_time:.2f}s")
    print(f"Speedup over greedy decoding: {greedy_time / spec_time:.2f}x")
    print(f"Speedup over greedy decoding with kv cache: {cache_time / spec_time:.2f}x")
    print(f"Metrics: {metrics}")
