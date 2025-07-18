from __future__ import annotations

"""Reinforcement learning based prompt optimizer, with advanced algorithmic and ethical assurance."""

import random
import logging
import re # For cleaning generated text (though primarily used in base PromptOptimizer, good to have)
from typing import Callable, Dict, Optional, List, Any

# These imports are assumed to work correctly from the user's setup as per the last instruction
from .prompt_optimizer import PromptOptimizer # Base class providing LLM, scoring, safety, and variation generation


# --- ANSI Escape Codes for Colors and Styles (for clear console output) ---
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
log.setLevel(logging.INFO) # Set default level to INFO
if not log.handlers: # Prevent adding multiple handlers if run multiple times
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    log.addHandler(console_handler)


# --- Advanced PromptRLOptimizer ---

class PromptRLOptimizer(PromptOptimizer):
    """
    An advanced Reinforcement Learning (RL) based prompt optimizer, built upon a Q-learning algorithm.
    This optimizer iteratively refines prompts by exploring a diverse space of variations,
    learning optimal prompt structures through a reward mechanism that rigorously incorporates
    ethical guidelines and prompt effectiveness.

    It aims to find prompts that not only elicit high-quality responses from an LLM but
    also adhere to safety and relevance standards, making it a robust and ethically-aware
    algorithmic component in prompt engineering workflows.
    """

    def __init__(
        self,
        model_name: str, # The name of the underlying language model for PromptOptimizer
        reward_fn: Callable[[str], float] | None = None, # Optional external reward function
        *,
        episodes: int = 10, # Number of RL episodes for training iterations
        epsilon: float = 0.2, # Epsilon for epsilon-greedy exploration (0.0 to 1.0)
        lr: float = 0.3, # Learning rate for Q-value updates
        device: Optional[str] = None, # Device to run models on ('cuda' or 'cpu')
    ) -> None:
        """
        Initializes the PromptRLOptimizer.

        Args:
            model_name (str): The name of the underlying language model to be used by the
                              base PromptOptimizer for generating variations, scoring, etc.
            reward_fn (Callable[[str], float] | None): An optional external reward function.
                                                        If None, the optimizer uses the negated
                                                        `self.score_prompt` from the base class
                                                        as its reward signal (higher reward is better).
            episodes (int): The number of episodes the RL agent will run to optimize the prompt.
            epsilon (float): The probability of choosing a random action (exploration) instead of
                             the best-known action (exploitation) during the Q-learning process.
            lr (float): The learning rate, determining how much new information (rewards) updates
                        the existing Q-values.
            device (Optional[str]): The computational device ('cuda' for GPU, 'cpu' for CPU)
                                    to load the language models on. Inherited by the base class.
        """
        log.info(f"{Colors.BLUE}Initializing PromptRLOptimizer for model: '{model_name}'...{Colors.RESET}")
        
        # Call super().__init__ to set up the base PromptOptimizer, which includes
        # the LanguageModelWrapper (for generate/score) and the safety_checker.
        super().__init__(model_name, device=device) 
        
        self.reward_fn = reward_fn
        self.episodes = episodes
        self.epsilon = epsilon
        self.lr = lr
        self._q_values: Dict[str, float] = {} # Stores the learned Q-values for each prompt (state) encountered

        log.info(f"{Colors.BLUE}PromptRLOptimizer initialized with episodes={episodes}, epsilon={epsilon}, lr={lr}.{Colors.RESET}")
        log.info(f"{Colors.DIM}  Reward function source: {'Custom (provided)' if reward_fn else 'Internal (negated score_prompt)'}{Colors.RESET}")


    def _reward(self, prompt: str) -> float:
        """
        Calculates the reward for a given prompt, central to the RL process.
        This function is designed with an explicit focus on ethical assurance:
        unsafe or irrelevant prompts receive a significant negative reward.

        Args:
            prompt (str): The prompt string for which to calculate the reward.

        Returns:
            float: The calculated reward value. Higher values indicate a more desirable prompt.
                   A very large negative value (`-1000.0`) is returned if the prompt fails
                   the ethical or relevance checks, strongly penalizing such prompts.
        """
        # --- Ethical and Relevance Check (Primary Filter for Reward) ---
        # This leverages the _is_safe_and_relevant method from the base PromptOptimizer.
        if not self._is_safe_and_relevant(prompt):
            log.warning(f"{Colors.YELLOW}RL Reward: Prompt '{prompt[:70]}...' failed ethical/relevance check. Assigning very low reward (-1000.0).{Colors.RESET}")
            return -1000.0 # Assign a strong, negative penalty for unsafe or irrelevant content

        # --- Calculate Intrinsic Reward (based on provided function or internal scoring) ---
        try:
            if self.reward_fn is not None:
                # If an external reward function is provided, use it directly.
                # This allows users to define custom metrics (e.g., task-specific performance).
                reward = self.reward_fn(prompt)
                log.debug(f"{Colors.DIM}RL Reward: External reward_fn returned {reward:.2f} for '{prompt[:50]}...'{Colors.RESET}")
            else:
                # Default reward: Use the base class's score_prompt method.
                # Since score_prompt typically returns lower values for better prompts, we negate it
                # to convert it into a reward signal where higher is better.
                # The score_prompt method itself already includes its own safety/relevance check and penalty.
                score = self.score_prompt(prompt) 
                reward = -score
                log.debug(f"{Colors.DIM}RL Reward: Internal score_prompt ({score:.2f}) converted to reward ({reward:.2f}) for '{prompt[:50]}...'{Colors.RESET}")
            
            return reward

        except Exception as e:
            # Catch any errors during reward calculation to prevent crashing the RL process
            log.error(f"{Colors.RED}RL Reward: Error calculating reward for prompt '{prompt[:50]}...': {e}. Assigning a default penalty (-500.0).{Colors.RESET}")
            return -500.0 # A moderate penalty if reward calculation itself fails unexpectedly

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """
        Optimizes a base prompt using a reinforcement learning approach (Q-learning).
        This method orchestrates the RL episodes, managing exploration, exploitation,
        Q-value updates, and ensuring all considered prompts meet ethical and relevance standards.

        The RL agent learns by iteratively:
        1. Generating and evaluating prompt variations.
        2. Receiving rewards based on the quality and ethical compliance of these variations.
        3. Updating its internal knowledge (Q-values) to favor more effective and safe prompts.

        Args:
            base_prompt (str): The initial prompt from which to start the optimization.
            n_variations (int): The number of prompt variations the optimizer should
                                generate and explore within each iteration of an episode.

        Returns:
            str: The final best-optimized prompt found after completing all RL episodes,
                 guaranteed to be ethically safe and relevant.

        Raises:
            RuntimeError: If the initial base prompt is unsafe, or if no safe and relevant
                          prompts can be generated or found during the entire optimization process.
        """
        log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}--- Starting RL-based Prompt Optimization for: '{base_prompt[:100]}...' ---{Colors.RESET}")

        # --- Initial Validation of the Base Prompt ---
        # It's crucial that the starting point is ethically sound.
        if not self._is_safe_and_relevant(base_prompt):
            log.critical(f"{Colors.RED}Initial base prompt '{base_prompt}' failed ethical/relevance check. Aborting RL optimization.{Colors.RESET}")
            raise RuntimeError("Base prompt is unsafe or irrelevant, cannot proceed with RL optimization.")

        # --- Initialize Q-values with the base prompt and its initial variations ---
        log.info(f"{Colors.BLUE}Generating initial prompt variations for RL state space...{Colors.RESET}")
        initial_raw_candidates = self.generate_variations(base_prompt, n_variations=n_variations)
        
        # Filter for safety and relevance. Only safe prompts enter the Q-value dictionary.
        safe_prompts_for_q_values: List[str] = [base_prompt] # Always include the safe base prompt
        for c in initial_raw_candidates:
            if self._is_safe_and_relevant(c):
                if c not in self._q_values: # Avoid re-adding duplicates
                    safe_prompts_for_q_values.append(c)
                    self._q_values[c] = self._reward(c) # Initialize Q-value for new safe candidates
            else:
                log.warning(f"{Colors.YELLOW}  Initial generated variation '{c[:70]}...' failed ethical/relevance check. Skipping for RL.{Colors.RESET}")

        if not safe_prompts_for_q_values:
            log.critical(f"{Colors.RED}No safe and relevant initial prompts or variations found after initial filtering. RL optimization cannot proceed.{Colors.RESET}")
            raise RuntimeError("No safe or relevant prompts available for RL initialization.")

        # Start the RL process with a randomly chosen safe prompt from the initial pool
        current_prompt = random.choice(safe_prompts_for_q_values)
        log.info(f"{Colors.GREEN}RL Optimization starting with initial prompt: '{current_prompt[:80]}...'{Colors.RESET}")
        log.info(f"{Colors.DIM}Current Q-value pool size: {len(self._q_values)}{Colors.RESET}")


        # --- Main Reinforcement Learning Loop (Episodes) ---
        for episode in range(self.episodes):
            log.info(f"\n{Colors.CYAN}--- RL Episode {episode + 1}/{self.episodes} ---{Colors.RESET}")
            
            chosen_prompt: str
            # Epsilon-greedy action selection: balance between exploration and exploitation
            if random.random() < self.epsilon:
                # EXPLORATION: Generate new variations or randomly pick from known safe prompts
                log.info(f"{Colors.DIM}  Action: Exploring new prompt variations (Epsilon-greedy exploration).{Colors.RESET}")
                
                # Generate new variations based on the current best prompt or a strategically chosen prompt.
                # Using current_prompt as the seed for new variations.
                exploration_raw_candidates = self.generate_variations(current_prompt, n_variations=n_variations)
                
                safe_exploration_candidates = []
                for ec in exploration_raw_candidates:
                    if ec not in self._q_values and self._is_safe_and_relevant(ec): # Only add new, safe candidates
                        safe_exploration_candidates.append(ec)
                        self._q_values[ec] = self._reward(ec) # Initialize Q-value for newly discovered safe states
                    elif not self._is_safe_and_relevant(ec):
                        log.warning(f"{Colors.YELLOW}  Exploration variation '{ec[:60]}...' failed safety check. Discarding.{Colors.RESET}")
                
                if safe_exploration_candidates:
                    chosen_prompt = random.choice(safe_exploration_candidates)
                    log.info(f"{Colors.DIM}  Chosen for exploration (new): '{chosen_prompt[:80]}...'{Colors.RESET}")
                elif self._q_values: # Fallback: if no new safe prompts, pick randomly from existing safe Q-value pool
                    chosen_prompt = random.choice(list(self._q_values.keys()))
                    log.info(f"{Colors.YELLOW}  No new safe exploration variations generated. Chosen randomly from known safe prompts: '{chosen_prompt[:80]}...'{Colors.RESET}")
                else:
                    log.critical(f"{Colors.RED}  RL failed to find any safe prompts even during exploration. Aborting.{Colors.RESET}")
                    raise RuntimeError("RL optimization failed: No safe prompts available for exploration.")
            else:
                # EXPLOITATION: Choose the prompt with the highest current Q-value
                log.info(f"{Colors.DIM}  Action: Exploiting best known prompts (Epsilon-greedy exploitation).{Colors.RESET}")
                if not self._q_values:
                    log.warning(f"{Colors.YELLOW}  No Q-values available for exploitation. Falling back to base prompt.{Colors.RESET}")
                    chosen_prompt = base_prompt # Critical fallback if Q-values somehow become empty
                else:
                    # Select the prompt (state) with the maximum Q-value
                    chosen_prompt = max(self._q_values, key=lambda p: self._q_values.get(p, -float('inf')))
                log.info(f"{Colors.DIM}  Chosen for exploitation: '{chosen_prompt[:80]}...' (Q-value: {self._q_values.get(chosen_prompt, 0.0):.2f}){Colors.RESET}")

            # --- Evaluate the Chosen Prompt and Update Q-value ---
            reward = self._reward(chosen_prompt) # Get reward, which includes ethical checks

            # Q-value update rule (simple form of Q-learning for a non-episodic/single-step context)
            old_q_value = self._q_values.get(chosen_prompt, 0.0) # Get current Q-value, default to 0 if new
            
            # Q(s,a) = Q(s,a) + lr * (reward - Q(s,a))
            # This updates the Q-value based on the immediate reward and the prediction error.
            self._q_values[chosen_prompt] = old_q_value + self.lr * (reward - old_q_value)
            
            log.info(f"{Colors.GREEN}  Prompt: '{chosen_prompt[:70]}...' | Reward: {reward:.2f} | Q-Value Updated: {old_q_value:.2f} -> {self._q_values[chosen_prompt]:.2f}{Colors.RESET}")

            # Update `current_prompt` for the next episode. It's usually the one with the highest Q-value overall.
            if self._q_values: # Ensure Q-values are not empty
                current_prompt = max(self._q_values, key=lambda p: self._q_values.get(p, -float('inf')))
                log.info(f"{Colors.DIM}  Current overall best prompt (for next episode): '{current_prompt[:80]}...' (Overall Best Q: {self._q_values.get(current_prompt, 0.0):.2f}){Colors.RESET}")
            else:
                # Fallback if Q-values become empty due to aggressive filtering or errors
                current_prompt = base_prompt
                log.warning(f"{Colors.YELLOW}  Q-value dictionary became empty. Resetting current_prompt to base prompt.{Colors.RESET}")


        # --- Final Selection of the Best Prompt ---
        # After all episodes, the prompt with the highest accumulated Q-value is the best.
        if not self._q_values:
            log.critical(f"{Colors.RED}No Q-values were established during RL optimization or all were discarded. Cannot determine best prompt.{Colors.RESET}")
            raise RuntimeError("RL optimization failed to establish Q-values or find a valid best prompt.")

        best_prompt_overall = max(self._q_values, key=lambda p: self._q_values.get(p, -float('inf')))
        final_q_value = self._q_values[best_prompt_overall]

        log.info(f"\n{Colors.GREEN}{Colors.BOLD}--- RL Optimization Complete ---{Colors.RESET}")
        log.info(f"{Colors.GREEN}🌟 Final Selected Prompt: '{best_prompt_overall}'{Colors.RESET}")
        log.info(f"{Colors.GREEN}🏆 Highest Q-Value: {final_q_value:.2f}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}---------------------------------------------------------------------------------{Colors.RESET}\n")

        return best_prompt_overall


# --- Example Usage (main block for testing) ---
# This block assumes `PromptOptimizer` and `LanguageModelWrapper` are correctly
# imported and functional from their respective files as dependencies.
if __name__ == "__main__":
    import torch # Required for device detection
    import os # For setting logging level from environment variable

    # Set logging level for a more verbose output during testing
    # os.environ['LOG_LEVEL'] = 'DEBUG' # Uncomment for very detailed logs
    log.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())


    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║            🧠 PromptRLOptimizer - Advanced Demonstration                 ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    # Choose a small, efficient model for demonstration.
    # 'distilgpt2' is good for quick local testing. For more sophisticated
    # prompt optimization, consider larger models like 'gpt2', 'microsoft/DialoGPT-medium',
    # or other open-source alternatives if computational resources permit.
    model_to_use = 'distilgpt2' 
    log.info(f"{Colors.BRIGHT_YELLOW}Using model: '{model_to_use}' for prompt generation and scoring.{Colors.RESET}")

    # --- Device Selection ---
    if torch.cuda.is_available():
        optimizer_device = 'cuda'
        log.info(f"{Colors.BRIGHT_GREEN}CUDA detected. Attempting to use GPU.{Colors.RESET}")
    else:
        optimizer_device = 'cpu'
        log.info(f"{Colors.BRIGHT_YELLOW}CUDA not available. Using CPU.{Colors.RESET}")

    try:
        # Example 1: Basic RL Optimization with default reward function
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- Example 1: Basic PromptRLOptimizer Usage ---{Colors.RESET}")
        # episodes and n_variations are kept low for quick demonstration
        rl_optimizer_1 = PromptRLOptimizer(model_name=model_to_use, device=optimizer_device, episodes=3, epsilon=0.5, lr=0.2)
        base_prompt_1 = "Generate a compelling short story about artificial intelligence and human creativity."
        optimized_prompt_1 = rl_optimizer_1.optimize_prompt(base_prompt_1, n_variations=3)
        
        log.info(f"\n{Colors.MAGENTA}Original Prompt: {base_prompt_1}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Final RL-Optimized Prompt: {optimized_prompt_1}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # Example 2: RL Optimization with a custom external reward function
        # This demonstrates how the reward signal can be customized based on external
        # criteria beyond just the internal LLM score, e.g., evaluating the generated
        # output quality based on specific task metrics.
        def custom_reward_func(prompt: str) -> float:
            """
            A mock custom reward function for demonstration.
            In a real scenario, this would evaluate the output generated by the prompt
            against a specific task or rubric.
            """
            # Simulate a reward that prefers longer, more detailed, and ethically aware prompts.
            reward_val = len(prompt) / 20.0 # Reward for length
            
            if "creativity" in prompt.lower() and "ethical implications" in prompt.lower():
                reward_val += 7.0 # Bonus for hitting key complex concepts
            if "bias" in prompt.lower() or "fairness" in prompt.lower():
                reward_val += 3.0 # Bonus for explicitly mentioning ethical concerns

            # Important: The _reward method of PromptRLOptimizer will internally call
            # _is_safe_and_relevant before applying this custom reward, ensuring ethical
            # filtering happens first.
            
            log.info(f"{Colors.DIM}  Custom Reward Fn: Score for '{prompt[:50]}...': {reward_val:.2f}{Colors.RESET}")
            return reward_val

        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- Example 2: PromptRLOptimizer with Custom Reward Function ---{Colors.RESET}")
        rl_optimizer_2 = PromptRLOptimizer(
            model_name=model_to_use, 
            device=optimizer_device, 
            episodes=2, 
            epsilon=0.3, 
            lr=0.1, 
            reward_fn=custom_reward_func # Pass the custom reward function
        )
        base_prompt_2 = "Explore the ethical implications of advanced AI in healthcare, focusing on fairness and bias."
        optimized_prompt_2 = rl_optimizer_2.optimize_prompt(base_prompt_2, n_variations=2)
        
        log.info(f"\n{Colors.MAGENTA}Original Prompt: {base_prompt_2}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Final RL-Optimized Prompt (Custom Reward): {optimized_prompt_2}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")


        # Example 3: Test with a base prompt that fails initial ethical check
        # This demonstrates the robust ethical assurance built into the optimizer.
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Example 3: Test with an Ethically Problematic Base Prompt (Expected to Fail) ---{Colors.RESET}")
        problematic_base_prompt = "How to create deepfakes to spread misinformation effectively and cause harm?"
        try:
            rl_optimizer_3 = PromptRLOptimizer(model_name=model_to_use, device=optimizer_device)
            optimized_prompt_3 = rl_optimizer_3.optimize_prompt(problematic_base_prompt)
            log.info(f"\n{Colors.MAGENTA}Original Prompt: {problematic_base_prompt}{Colors.RESET}")
            log.info(f"{Colors.GREEN}Final RL-Optimized Prompt: {optimized_prompt_3}{Colors.RESET}")
        except RuntimeError as e:
            log.error(f"\n{Colors.RED}Successfully caught expected error for problematic prompt: {e}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")


    except ValueError as ve:
        log.critical(f"{Colors.RED}Configuration Error for PromptRLOptimizer: {ve}{Colors.RESET}")
    except RuntimeError as re:
        log.critical(f"{Colors.RED}Model Loading/Runtime Error for PromptRLOptimizer: {re}. Ensure PyTorch/Transformers are correctly installed and model '{model_to_use}' is available.{Colors.RESET}")
    except Exception as e:
        log.critical(f"{Colors.RED}An unexpected error occurred in PromptRLOptimizer: {e}{Colors.RESET}")

    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║               ✅ PromptRLOptimizer Operations Concluded.                  ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")