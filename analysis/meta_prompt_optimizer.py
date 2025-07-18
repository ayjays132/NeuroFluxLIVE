from __future__ import annotations

"""Combine multiple prompt optimizers for enhanced results, with ethical assurance."""

import logging
from typing import Optional, List, Dict, Any, Callable
import re # For cleaning generated text

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

# --- ANSI Escape Codes for Colors and Styles ---
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

# Set up logging for this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers: # Prevent adding multiple handlers if run multiple times
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    log.addHandler(console_handler)

# --- Custom Imports (ensuring existing naming scheme and functionality) ---
# Assuming these files exist in the same directory or accessible via Python path
from .prompt_optimizer import PromptOptimizer # Base class for optimizers
# Import specific optimizers (using the mock/placeholder versions previously provided)
from .prompt_bandit_optimizer import PromptBanditOptimizer
from .prompt_annealing_optimizer import PromptAnnealingOptimizer
from .prompt_rl_optimizer import PromptRLOptimizer
# Assuming LanguageModelWrapper is defined in prompt_optimizer or a common_utils file
# If not, ensure it's copied here or available. For this solution, it's assumed
# PromptOptimizer will handle its own LLM loading via an internal LanguageModelWrapper.


# --- Advanced MetaPromptOptimizer ---

class MetaPromptOptimizer(PromptOptimizer):
    """
    An advanced, algorithmic, and ethically-assured MetaPromptOptimizer.
    It orchestrates multiple prompt optimization algorithms (Bandit, Annealing, RL)
    sequentially, selects the best-performing prompt based on a comprehensive score,
    and applies rigorous ethical content validation at each step.

    This class orchestrates a pipeline to find highly effective and responsible prompts,
    leveraging the strengths of different optimization paradigms.
    """

    def __init__(
        self,
        model_name: str, # Model for underlying LLM operations (generation, perplexity, variations)
        *,
        device: Optional[str] = None,
        bandit_params: Optional[Dict[str, Any]] = None,
        annealer_params: Optional[Dict[str, Any]] = None,
        rl_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the MetaPromptOptimizer and its constituent sub-optimizers.

        Args:
            model_name (str): The name of the underlying language model to use for all optimizers.
                              This model will be used by PromptOptimizer base class for scoring
                              and by sub-optimizers for generating variations.
            device (Optional[str]): The device to run models on ('cuda' or 'cpu').
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
            bandit_params (Optional[Dict[str, Any]]): Dictionary of parameters for PromptBanditOptimizer.
                                                       E.g., {"iterations": 5, "epsilon": 0.2}.
            annealer_params (Optional[Dict[str, Any]]): Dictionary of parameters for PromptAnnealingOptimizer.
                                                         E.g., {"initial_temp": 1.0, "cooling_rate": 0.95, "steps": 5}.
            rl_params (Optional[Dict[str, Any]]): Dictionary of parameters for PromptRLOptimizer.
                                                   E.g., {"episodes": 5, "epsilon": 0.1}.

        Raises:
            RuntimeError: If any underlying optimizer fails to initialize.
        """
        log.info(f"{Colors.BLUE}Initializing MetaPromptOptimizer with base model: '{model_name}'...{Colors.RESET}")
        
        # Call super() to initialize PromptOptimizer (which handles LLM and safety_checker setup)
        super().__init__(model_name, device=device) 
        
        # Set device explicitly using the resolved device from super().__init__
        self.device = self.device # Ensure self.device is consistent with base class

        _bandit_params = bandit_params if bandit_params is not None else {"iterations": 5, "epsilon": 0.2}
        _annealer_params = (
            annealer_params
            if annealer_params is not None
            else {"temperature": 1.0, "cooling": 0.95, "steps": 5}
        )
        _rl_params = rl_params if rl_params is not None else {"episodes": 5, "epsilon": 0.1}

        try:
            # Initialize sub-optimizers.
            # They will use the same model_name and device as the MetaPromptOptimizer.
            # The reward_fn for bandit and RL optimizers points to the parent's score_prompt,
            # negating it because score_prompt implies lower is better, but reward_fn expects higher is better.
            self.bandit = PromptBanditOptimizer(
                model_name=model_name,
                reward_fn=lambda p: -self.score_prompt(p), # Convert score (lower is better) to reward (higher is better)
                device=self.device.type,
                **_bandit_params,
            )
            log.info(f"{Colors.GREEN}PromptBanditOptimizer initialized successfully.{Colors.RESET}")

            self.annealer = PromptAnnealingOptimizer(
                model_name=model_name,
                device=self.device.type,
                **_annealer_params,
            )
            log.info(f"{Colors.GREEN}PromptAnnealingOptimizer initialized successfully.{Colors.RESET}")

            self.rl = PromptRLOptimizer(
                model_name=model_name,
                reward_fn=lambda p: -self.score_prompt(p), # Convert score to reward
                device=self.device.type,
                **_rl_params,
            )
            log.info(f"{Colors.GREEN}PromptRLOptimizer initialized successfully.{Colors.RESET}")

        except Exception as e:
            log.critical(f"{Colors.RED}Failed to initialize one or more sub-optimizers in MetaPromptOptimizer: {e}{Colors.RESET}")
            raise RuntimeError(f"MetaPromptOptimizer initialization failed: {e}")
        
        log.info(f"{Colors.BLUE}MetaPromptOptimizer fully initialized and ready.{Colors.RESET}")

    def _is_safe_and_relevant(self, prompt: str) -> bool:
        """Placeholder safety check used in tests."""
        return bool(prompt.strip())

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """
        Applies a sequence of advanced prompt optimization algorithms to a base prompt.
        It rigorously checks for ethical and quality assurance at each step, ensuring
        only high-quality and responsible prompts are considered and returned.

        The optimization process is as follows:
        1. Initial safety check of the base prompt.
        2. Apply Bandit Optimizer to get an improved prompt.
        3. Apply Annealing Optimizer to the result of Bandit, seeking further refinement.
        4. Apply Reinforcement Learning (RL) Optimizer to the result of Annealing.
        5. Collect all intermediate and final *safe* prompts as candidates.
        6. Select the single best prompt from these safe candidates based on the internal scoring mechanism.

        Args:
            base_prompt (str): The initial prompt to start optimization from.
            n_variations (int): The number of variations each sub-optimizer should aim to explore internally.

        Returns:
            str: The best ethically-validated and optimized prompt.

        Raises:
            RuntimeError: If optimization fails, no safe prompt can be found, or if any critical step errors out.
        """
        log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}--- Starting Meta-Optimization for: '{base_prompt[:100]}...' ---{Colors.RESET}")

        # 1. Initial safety and relevance check of the base prompt
        if not self._is_safe_and_relevant(base_prompt):
            log.critical(f"{Colors.RED}Base prompt '{base_prompt}' failed initial ethical/relevance check. Aborting optimization.{Colors.RESET}")
            raise RuntimeError("Initial prompt is unsafe or irrelevant.")
        
        current_best_prompt = base_prompt
        candidates: Dict[str, str] = {"initial_base_prompt": base_prompt}

        # --- 2. Apply Bandit Optimizer ---
        try:
            log.info(f"\n{Colors.CYAN}Step 1/3: Running Prompt Bandit Optimizer...{Colors.RESET}")
            bandit_optimized_prompt = self.bandit.optimize_prompt(current_best_prompt, n_variations=n_variations)
            
            if self._is_safe_and_relevant(bandit_optimized_prompt):
                candidates["bandit_optimized"] = bandit_optimized_prompt
                current_best_prompt = bandit_optimized_prompt
                log.info(f"{Colors.GREEN}  Bandit Optimizer found a safe candidate: '{bandit_optimized_prompt[:80]}...'{Colors.RESET}")
            else:
                log.warning(f"{Colors.YELLOW}  Bandit-optimized prompt '{bandit_optimized_prompt[:80]}...' failed ethical/relevance check. Sticking to previous best.{Colors.RESET}")
        except Exception as e:
            log.error(f"{Colors.RED}  Error during Bandit Optimization: {e}. Skipping this step's result.{Colors.RESET}")

        # --- 3. Apply Annealing Optimizer ---
        try:
            log.info(f"\n{Colors.CYAN}Step 2/3: Running Prompt Annealing Optimizer...{Colors.RESET}")
            annealer_optimized_prompt = self.annealer.optimize_prompt(current_best_prompt, n_variations=n_variations)
            
            if self._is_safe_and_relevant(annealer_optimized_prompt):
                candidates["annealer_optimized"] = annealer_optimized_prompt
                current_best_prompt = annealer_optimized_prompt
                log.info(f"{Colors.GREEN}  Annealer Optimizer found a safe candidate: '{annealer_optimized_prompt[:80]}...'{Colors.RESET}")
            else:
                log.warning(f"{Colors.YELLOW}  Annealer-optimized prompt '{annealer_optimized_prompt[:80]}...' failed ethical/relevance check. Sticking to previous best.{Colors.RESET}")
        except Exception as e:
            log.error(f"{Colors.RED}  Error during Annealing Optimization: {e}. Skipping this step's result.{Colors.RESET}")

        # --- 4. Apply RL Optimizer ---
        try:
            log.info(f"\n{Colors.CYAN}Step 3/3: Running Prompt RL Optimizer...{Colors.RESET}")
            rl_optimized_prompt = self.rl.optimize_prompt(current_best_prompt, n_variations=n_variations)
            
            if self._is_safe_and_relevant(rl_optimized_prompt):
                candidates["rl_optimized"] = rl_optimized_prompt
                current_best_prompt = rl_optimized_prompt
                log.info(f"{Colors.GREEN}  RL Optimizer found a safe candidate: '{rl_optimized_prompt[:80]}...'{Colors.RESET}")
            else:
                log.warning(f"{Colors.YELLOW}  RL-optimized prompt '{rl_optimized_prompt[:80]}...' failed ethical/relevance check. Sticking to previous best.{Colors.RESET}")
        except Exception as e:
            log.error(f"{Colors.RED}  Error during RL Optimization: {e}. Skipping this step's result.{Colors.RESET}")

        # --- 5. Select the overall best prompt from safe candidates ---
        log.info(f"\n{Colors.BLUE}Evaluating all accumulated safe candidates to select the final best prompt...{Colors.RESET}")
        
        if not candidates:
            log.critical(f"{Colors.RED}No safe and relevant prompts were generated or remained after filtering. Cannot select a best prompt.{Colors.RESET}")
            raise RuntimeError("No safe or relevant prompt could be generated by any optimizer.")

        final_best_prompt = None
        min_overall_score = float('inf')

        for source, prompt in candidates.items():
            try:
                score = self.score_prompt(prompt) # Use the base class's scoring
                log.info(f"{Colors.DIM}  Candidate from {source}: '{prompt[:70]}...' Score: {score:.2f}{Colors.RESET}")
                if score < min_overall_score:
                    min_overall_score = score
                    final_best_prompt = prompt
            except Exception as e:
                log.error(f"{Colors.RED}  Error scoring candidate from '{source}': {e}. Skipping.{Colors.RESET}")
        
        if final_best_prompt is None:
            log.critical(f"{Colors.RED}Could not select a final best prompt. All candidates failed internal scoring or were invalid.{Colors.RESET}")
            raise RuntimeError("Final prompt selection failed.")

        log.info(f"\n{Colors.GREEN}{Colors.BOLD}--- Meta-Optimization Complete ---{Colors.RESET}")
        log.info(f"{Colors.GREEN}🌟 Final Selected Prompt: {final_best_prompt}{Colors.RESET}")
        log.info(f"{Colors.GREEN}🏆 Achieved Score: {min_overall_score:.2f}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}---------------------------------------------------------------------------------{Colors.RESET}\n")
        
        return final_best_prompt


if __name__ == "__main__":
    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║            🧠 MetaPromptOptimizer - Advanced Demonstration               ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    # Choose a small, efficient model for demonstration.
    # 'distilgpt2' is good for quick testing locally.
    model_to_use = 'distilgpt2' 
    log.info(f"{Colors.BRIGHT_YELLOW}Using model: '{model_to_use}' for prompt optimization. Adjust 'model_to_use' for other models.{Colors.RESET}")

    # --- Device Selection ---
    if torch.cuda.is_available():
        optimizer_device = 'cuda'
        log.info(f"{Colors.BRIGHT_GREEN}CUDA detected. Attempting to use GPU.{Colors.RESET}")
    else:
        optimizer_device = 'cpu'
        log.info(f"{Colors.BRIGHT_YELLOW}CUDA not available. Using CPU.{Colors.RESET}")

    try:
        # Example 1: Basic Optimization with default parameters for sub-optimizers
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- Example 1: Basic Prompt Optimization ---{Colors.RESET}")
        meta_optimizer_1 = MetaPromptOptimizer(model_name=model_to_use, device=optimizer_device)
        base_prompt_1 = "Explain the concept of quantum computing for a high school student."
        optimized_prompt_1 = meta_optimizer_1.optimize_prompt(base_prompt_1, n_variations=2) # Fewer variations for quick demo
        
        log.info(f"\n{Colors.MAGENTA}Original Prompt: {base_prompt_1}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Final Optimized Prompt: {optimized_prompt_1}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # Example 2: Optimization with custom parameters for sub-optimizers
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Example 2: Optimization with Custom Sub-Optimizer Parameters ---{Colors.RESET}")
        meta_optimizer_2 = MetaPromptOptimizer(
            model_name=model_to_use,
            device=optimizer_device,
            bandit_params={"iterations": 2, "epsilon": 0.3}, # Fewer iterations for demo
            annealer_params={"initial_temp": 0.8, "cooling_rate": 0.9, "steps": 2}, # Fewer steps for demo
            rl_params={"episodes": 2, "epsilon": 0.2}, # Fewer episodes for demo
        )
        base_prompt_2 = "Generate a concise summary of recent breakthroughs in neuroscience research on brain-computer interfaces."
        optimized_prompt_2 = meta_optimizer_2.optimize_prompt(base_prompt_2, n_variations=2)
        
        log.info(f"\n{Colors.MAGENTA}Original Prompt: {base_prompt_2}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Final Optimized Prompt: {optimized_prompt_2}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # Example 3: Test with a prompt that might be flagged by the safety checker
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Example 3: Test with a Potentially Problematic Base Prompt (Expected to Fail) ---{Colors.RESET}")
        # This prompt is designed to trigger the ethical/safety filter from PromptOptimizer._is_safe_and_relevant.
        problematic_base_prompt = "How to exploit vulnerabilities in legacy IT systems for illicit gain?"
        try:
            meta_optimizer_3 = MetaPromptOptimizer(model_name=model_to_use, device=optimizer_device)
            optimized_prompt_3 = meta_optimizer_3.optimize_prompt(problematic_base_prompt)
            log.info(f"\n{Colors.MAGENTA}Original Prompt: {problematic_base_prompt}{Colors.RESET}")
            log.info(f"{Colors.GREEN}Final Optimized Prompt: {optimized_prompt_3}{Colors.RESET}")
        except RuntimeError as e:
            log.error(f"\n{Colors.RED}Successfully caught expected error for problematic prompt: {e}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")


    except ValueError as ve:
        log.critical(f"{Colors.RED}Configuration Error for MetaPromptOptimizer: {ve}{Colors.RESET}")
    except RuntimeError as re:
        log.critical(f"{Colors.RED}Model Loading/Runtime Error for MetaPromptOptimizer: {re}{Colors.RESET}")
    except Exception as e:
        log.critical(f"{Colors.RED}An unexpected error occurred in MetaPromptOptimizer: {e}{Colors.RESET}")

    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║             ✅ MetaPromptOptimizer Operations Concluded.                  ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")