# SelfResearch/optimization/prompt_optimizer.py

from __future__ import annotations

import math
import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- ANSI Escape Codes for Colors and Styles (for consistent logging) ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    INVERT = "\033[7m"
    HIDDEN = "\033[8m"
    STRIKETHROUGH = "\033[9m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

# --- Configure Logging for structured and colored output ---
class ColoredFormatter(logging.Formatter):
    FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
    
    LOG_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED + Colors.BOLD,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD + Colors.UNDERLINE
    }

    def format(self, record):
        log_fmt = self.LOG_COLORS.get(record.levelno) + self.FORMAT + Colors.RESET
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    log.addHandler(console_handler)


class PromptOptimizer:
    """Generate and score prompt variations to select high quality prompts."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        """
        Initializes the PromptOptimizer.

        Args:
            model_name (str): The name of the pre-trained language model to use
                              for generating prompt variations and calculating perplexity.
            device (Optional[str]): The device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        """
        log.info(f"{Colors.BLUE}Initializing PromptOptimizer with model: {model_name}...{Colors.RESET}")
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        try:
            log.info(f"{Colors.CYAN}Loading model '{model_name}' on {self.device}...{Colors.RESET}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval() # Set model to evaluation mode for consistent behavior

            # Configure text generation pipeline
            pipeline_device_id = 0 if self.device.type == "cuda" else -1
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device_id,
                truncation=True, # Enable truncation for robustness
                max_length=self.tokenizer.model_max_length, # Ensure max length is respected
            )
            log.info(f"{Colors.GREEN}PromptOptimizer initialized successfully on {self.device} with model {model_name}.{Colors.RESET}")
        except Exception as e:
            log.critical(f"{Colors.RED}Failed to load model or tokenizer for PromptOptimizer: {e}{Colors.RESET}")
            raise RuntimeError(f"PromptOptimizer initialization failed: {e}")

    def generate_variations(
        self, base_prompt: str, n_variations: int = 5, *, temperature: float = 1.0
    ) -> List[str]:
        """
        Generates prompt variations from a base prompt using the internal language model.

        Parameters
        ----------
        base_prompt (str):
            Prompt used as the starting point for generation.
        n_variations (int):
            Number of variations to produce.
        temperature (float):
            Sampling temperature controlling randomness of generations. Higher values lead to more diverse variations.

        Returns
        -------
        List[str]: A list of generated prompt variations.
        """
        log.info(f"{Colors.BLUE}Generating {n_variations} variations for base prompt: '{base_prompt[:80]}...'{Colors.RESET}")
        if not base_prompt.strip():
            log.warning(f"{Colors.YELLOW}Base prompt is empty. Returning base prompt only.{Colors.RESET}")
            return [base_prompt]

        outputs = self.generator(
            base_prompt,
            num_return_sequences=n_variations,
            max_new_tokens=20, # Generate a short continuation for variation
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id, # Prevents warnings/errors with batch generation
        )
        variations = []
        for i, out in enumerate(outputs):
            text = out["generated_text"]
            # Ensure the generated text starts with the base prompt and strip the base prompt to get the variation part
            if text.startswith(base_prompt):
                variation_part = text[len(base_prompt):].strip()
            else:
                variation_part = text.strip() # If it doesn't start, take the whole generated part

            # Only append if the variation part is meaningful
            if variation_part:
                variations.append(f"{base_prompt} {variation_part}")
                log.debug(f"{Colors.DIM}  Variant {i+1}: '{variations[-1][:100]}...'{Colors.RESET}")
            else:
                variations.append(base_prompt) # Fallback to just base prompt if no meaningful variation
                log.debug(f"{Colors.DIM}  Variant {i+1}: No new text generated, using base prompt.{Colors.RESET}")
        
        log.info(f"{Colors.GREEN}Generated {len(variations)} prompt variations.{Colors.RESET}")
        return variations

    def score_prompt(self, prompt: str) -> float:
        """
        Computes the perplexity of a prompt as a quality score. Lower perplexity generally indicates
        a more natural or "expected" sequence of words by the language model.

        Parameters
        ----------
        prompt (str): The prompt string to score.

        Returns
        -------
        float: The perplexity score. Lower is better. Returns float('inf') on error.
        """
        if not prompt.strip():
            log.warning(f"{Colors.YELLOW}Received empty prompt for scoring. Returning infinite perplexity.{Colors.RESET}")
            return float('inf')

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length).to(self.device)
            if inputs.input_ids.numel() == 0:
                log.warning(f"{Colors.YELLOW}Prompt '{prompt[:50]}...' tokenized to empty input IDs. Returning infinite perplexity.{Colors.RESET}")
                return float('inf')

            with torch.no_grad():
                # Calculate language modeling loss
                loss = self.model(**inputs, labels=inputs["input_ids"]).loss
            
            perplexity = float(math.exp(loss.item()))
            log.debug(f"{Colors.DIM}  Scored prompt '{prompt[:50]}...': Perplexity={perplexity:.2f}{Colors.RESET}")
            return perplexity
        except Exception as e:
            log.error(f"{Colors.RED}Error scoring prompt '{prompt[:50]}...': {e}. Returning infinite perplexity.{Colors.RESET}")
            return float('inf')

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """
        Generates prompt variations and returns the one with the lowest perplexity score.
        This is a basic optimization strategy.

        Parameters
        ----------
        base_prompt (str): The starting prompt for optimization.
        n_variations (int): The number of variations to generate and evaluate.

        Returns
        -------
        str: The optimized prompt with the lowest perplexity score.
        """
        log.info(f"{Colors.BLUE}Optimizing prompt '{base_prompt[:80]}...' using perplexity scoring...{Colors.RESET}")
        candidates = self.generate_variations(base_prompt, n_variations=n_variations)
        if not candidates:
            log.warning(f"{Colors.YELLOW}No candidates generated. Returning base prompt.{Colors.RESET}")
            return base_prompt

        # Ensure base_prompt is always considered as a candidate
        if base_prompt not in candidates:
            candidates.append(base_prompt)

        try:
            # Select the candidate with the minimum perplexity score
            best_prompt = min(candidates, key=self.score_prompt)
            log.info(f"{Colors.GREEN}Perplexity-optimized prompt: '{best_prompt[:100]}...'{Colors.RESET}")
            return best_prompt
        except Exception as e:
            log.error(f"{Colors.RED}Error during perplexity-based prompt optimization: {e}. Returning base prompt.{Colors.RESET}")
            return base_prompt

if __name__ == "__main__":
    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║                 🧪 PromptOptimizer Self-Test 🧪                         ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    # Example: Use a small, readily available model like "distilgpt2"
    optimizer_model_name = "distilgpt2"
    test_device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        optimizer = PromptOptimizer(model_name=optimizer_model_name, device=test_device)
        
        base_prompt_example = "Explain the concept of quantum entanglement in simple terms:"
        log.info(f"\n{Colors.BLUE}{Colors.BOLD}--- Testing generate_variations ---{Colors.RESET}")
        variations = optimizer.generate_variations(base_prompt_example, n_variations=3, temperature=0.8)
        for i, var in enumerate(variations):
            log.info(f"{Colors.CYAN}  Variation {i+1}: {var}{Colors.RESET}")

        log.info(f"\n{Colors.BLUE}{Colors.BOLD}--- Testing score_prompt ---{Colors.RESET}")
        sample_prompt_1 = "Quantum entanglement means two particles are linked."
        sample_prompt_2 = "This is a random sentence not related to physics."
        
        score1 = optimizer.score_prompt(sample_prompt_1)
        log.info(f"{Colors.MAGENTA}  Score for '{sample_prompt_1[:50]}...': {score1:.2f}{Colors.RESET}")
        
        score2 = optimizer.score_prompt(sample_prompt_2)
        log.info(f"{Colors.MAGENTA}  Score for '{sample_prompt_2[:50]}...': {score2:.2f}{Colors.RESET}")
        
        if score1 < score2:
            log.info(f"{Colors.GREEN}  (Expected: Lower perplexity for more coherent/relevant text){Colors.RESET}")
        else:
            log.warning(f"{Colors.YELLOW}  (Unexpected: Higher perplexity for more coherent/relevant text){Colors.RESET}")


        log.info(f"\n{Colors.BLUE}{Colors.BOLD}--- Testing optimize_prompt (Perplexity-based) ---{Colors.RESET}")
        optimized_p = optimizer.optimize_prompt(base_prompt_example, n_variations=5)
        log.info(f"{Colors.GREEN}  Optimized Prompt: {optimized_p}{Colors.RESET}")

    except RuntimeError as re:
        log.critical(f"{Colors.RED}Test failed due to initialization error: {re}{Colors.RESET}")
    except Exception as e:
        log.critical(f"{Colors.RED}An unexpected error occurred during test: {e}{Colors.RESET}")

    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║                     Test Complete.                                      ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")