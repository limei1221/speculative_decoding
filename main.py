import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Dict, Any, List


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    gamma: int = 5  # Number of draft tokens to generate
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    max_new_tokens: int = 128
    early_stop: bool = True
    verbose: bool = False
    max_iterations: int = 100  # Maximum iterations to prevent infinite loops


class SpeculativeDecoder:
    """Speculative decoding implementation."""

    def __init__(self, target_model, draft_model, tokenizer, config: SpeculativeConfig):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config

        self.device = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.target_model = self.target_model.to(self.device)
        self.draft_model = self.draft_model.to(self.device)

        self.target_model.eval()
        self.draft_model.eval()

    def _sample_from_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits using top-k and top-p sampling."""
        if self.config.do_sample:
            logits = logits / self.config.temperature

            if self.config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            tokens = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

        return tokens, probs

    def _generate_draft_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Generate draft tokens using the draft model."""
        with torch.no_grad():
            draft_outputs = self.draft_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.gamma,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )

            draft_tokens = draft_outputs.sequences[:, input_ids.size(1):]  # torch.Size([1, 5])
            draft_probs = torch.stack(draft_outputs.scores).softmax(-1).squeeze(1)  # torch.Size([5, 151936])

            return draft_tokens, draft_probs

    def _verify_draft_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                             draft_tokens: torch.Tensor, draft_probs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens using the target model with improved acceptance strategy."""
        with torch.no_grad():
            extended_input_ids = torch.cat([input_ids, draft_tokens], dim=1)
            extended_attention_mask = torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1)

            outputs = self.target_model(
                extended_input_ids,
                attention_mask=extended_attention_mask,
                return_dict=True,
            )

            target_logits = outputs.logits[:, input_ids.size(1) - 1:, :]  # torch.Size([1, 6, 151936])
            target_probs = F.softmax(target_logits, dim=-1)

            accepted_tokens = []
            num_accepted = 0

            for i in range(draft_tokens.size(1)):
                draft_token = draft_tokens[0, i]
                draft_prob = draft_probs[i, draft_token]
                target_prob = target_probs[0, i, draft_token]

                if target_prob > 0 and draft_prob > 0:
                    acceptance_prob = min(target_prob / draft_prob, 1.0)

                    if torch.rand(1, device=self.device).item() < acceptance_prob:
                        accepted_tokens.append(draft_token.item())
                        num_accepted += 1
                    else:
                        break
                else:
                    break

            next_token_logits = target_logits[0, num_accepted, :]
            next_token, _ = self._sample_from_logits(next_token_logits.unsqueeze(0))
            accepted_tokens.append(next_token.item())

            return torch.tensor([accepted_tokens], device=self.device), num_accepted

    def _adaptive_gamma(self, acceptance_ratio: float) -> int:
        """Adaptively adjust gamma based on acceptance ratio."""
        if acceptance_ratio > 0.8:
            return min(self.config.gamma + 1, 10)  # Increase gamma if acceptance is high
        elif acceptance_ratio < 0.3:
            return max(self.config.gamma - 1, 2)  # Decrease gamma if acceptance is low
        else:
            return self.config.gamma  # Keep current gamma

    def decode(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Main decoding function with speculative decoding and optimizations."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        input_len = input_ids.size(1)
        generated_tokens = []

        total_proposed = 0
        total_accepted = 0
        iterations = 0
        current_gamma = self.config.gamma

        while len(generated_tokens) < self.config.max_new_tokens:
            iterations += 1

            if self.config.verbose:
                print(f"Iteration {iterations}: Current length {len(generated_tokens)}, Gamma: {current_gamma}")

            draft_tokens, draft_probs = self._generate_draft_tokens(input_ids, attention_mask)

            accepted_tokens, num_accepted = self._verify_draft_tokens(
                input_ids, attention_mask, draft_tokens, draft_probs
            )

            total_proposed += draft_tokens.size(1)
            total_accepted += num_accepted

            for token in accepted_tokens[0]:
                generated_tokens.append(token)
                if token == self.tokenizer.eos_token_id and self.config.early_stop:
                    break

            if (self.tokenizer.eos_token_id in generated_tokens and self.config.early_stop) or len(
                generated_tokens) >= self.config.max_new_tokens:
                break

            if iterations > self.config.max_iterations:
                print("Warning: Maximum iterations reached")
                break

            input_ids = torch.cat([input_ids, accepted_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(accepted_tokens)], dim=1)

            if iterations > 1:
                acceptance_ratio = total_accepted / total_proposed
                current_gamma = self._adaptive_gamma(acceptance_ratio)

        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        metrics = {
            'total_proposed': total_proposed,
            'total_accepted': total_accepted,
            'acceptance_ratio': total_accepted / total_proposed if total_proposed > 0 else 0,
            'iterations': iterations,
            'generated_length': len(generated_tokens),
            'final_gamma': current_gamma,
            'efficiency': len(generated_tokens) / iterations if iterations > 0 else 0
        }

        return completion, metrics


def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Simple greedy decoding for comparison."""
    device = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.size(1)
    completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return completion


def benchmark_decoding(target_model, draft_model, tokenizer, prompt: str, config: SpeculativeConfig):
    """Benchmark different decoding strategies."""
    print("===BENCHMARKING DECODING STRATEGIES===")

    gamma_values = [3, 5, 7, 10]
    results = {}

    for gamma in gamma_values:
        print(f"\nTesting with gamma = {gamma}")
        config.gamma = gamma
        config.early_stop = False

        decoder = SpeculativeDecoder(target_model, draft_model, tokenizer, config)

        start_time = time.time()
        output, metrics = decoder.decode(prompt)
        end_time = time.time()

        results[gamma] = {
            'time': end_time - start_time,
            'output_length': len(output),
            'acceptance_ratio': metrics['acceptance_ratio'],
            'efficiency': metrics['efficiency']
        }

        print(f"  Time: {end_time - start_time:.3f}s")
        print(f"  Acceptance ratio: {metrics['acceptance_ratio']:.3f}")
        print(f"  Efficiency: {metrics['efficiency']:.3f}")

    optimal_gamma = max(results.keys(), key=lambda g: results[g]['efficiency'])
    print(f"\nOptimal gamma: {optimal_gamma}")

    return results, optimal_gamma


if __name__ == "__main__":
    print("Loading models...")
    model_name = "Qwen/Qwen3-1.7B"
    draft_model_name = "Qwen/Qwen3-0.6B"

    target_model = AutoModelForCausalLM.from_pretrained(model_name)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

    assert tokenizer.vocab_size == draft_tokenizer.vocab_size, "Vocabulary sizes must match"
    print(f"Models loaded successfully. Vocabulary size: {tokenizer.vocab_size}")

    prompt = "introduce yourself"
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    print(f"Prompt: {formatted_prompt}")

    config = SpeculativeConfig(
        gamma=5,
        temperature=0.6,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        max_new_tokens=128,
        early_stop=False,
        verbose=False,
    )

    benchmark_results, optimal_gamma = benchmark_decoding(
        target_model, draft_model, tokenizer, formatted_prompt, config
    )

    config.gamma = optimal_gamma
    config.early_stop = True

    decoder = SpeculativeDecoder(target_model, draft_model, tokenizer, config)

    print("Running greedy decoding...")
    start_time = time.time()
    greedy_output = greedy_decode(target_model, tokenizer, formatted_prompt, max_new_tokens=128)
    greedy_time = time.time() - start_time
    print(f"Greedy output: {greedy_output}")
    print(f"Greedy time: {greedy_time:.2f}s")

    print(f"\nRunning speculative decoding (gamma={optimal_gamma})...")
    start_time = time.time()
    spec_output, metrics = decoder.decode(formatted_prompt)
    spec_time = time.time() - start_time
    print(f"Speculative output: {spec_output}")
    print(f"Speculative time: {spec_time:.2f}s")
    print(f"Speedup: {greedy_time / spec_time:.2f}x")
    print(f"Metrics: {metrics}")
