from __future__ import annotations

"""Model wrapper utilities for HuggingFace models."""

import logging
from typing import Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils.colors import Colors

# Configure a basic logger for the module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    console_handler = logging.StreamHandler()
    # Using a simple formatter for this example, can be extended like in rubric_grader.py
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)


class LanguageModelWrapper:
    """
    An advanced, robust wrapper around HuggingFace causal language models.
    It supports flexible text generation strategies and provides detailed logging
    for better insights into model operations.
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        """
        Initializes the LanguageModelWrapper, loading the specified model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained model to load (e.g., 'gpt2', 'microsoft/DialoGPT-medium').
            device (Optional[str]): The device to load the model on ('cuda' or 'cpu').
                                    Defaults to 'cuda' if available, otherwise 'cpu'.

        Raises:
            ValueError: If an unsupported device string is provided.
            RuntimeError: If model or tokenizer loading fails.
        """
        log.info(f"{Colors.BLUE}Initializing LanguageModelWrapper for model: '{model_name}'...{Colors.RESET}")
        
        # --- Device Setup ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"{Colors.DIM}No device specified. Auto-detecting: {self.device.type.upper()} selected.{Colors.RESET}")
        elif device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device(device)
                log.info(f"{Colors.GREEN}{Colors.BOLD}Model will be loaded on GPU: {self.device}{Colors.RESET}")
            else:
                log.warning(f"{Colors.YELLOW}CUDA requested but not available. Falling back to CPU.{Colors.RESET}")
                self.device = torch.device('cpu')
        elif device == 'cpu':
            self.device = torch.device('cpu')
            log.info(f"{Colors.BLUE}Model will be loaded on CPU.{Colors.RESET}")
        else:
            raise ValueError(f"{Colors.RED}Unsupported device: '{device}'. Please choose 'cpu' or 'cuda'.{Colors.RESET}")

        # --- Model and Tokenizer Loading ---
        try:
            log.info(f"{Colors.CYAN}Loading tokenizer for '{model_name}'...{Colors.RESET}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Set pad_token_id to eos_token_id if not already set, common for generation
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            log.info(f"{Colors.GREEN}Tokenizer loaded.{Colors.RESET}")

            log.info(f"{Colors.CYAN}Loading model '{model_name}' and moving to {self.device}...{Colors.RESET}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval() # Set model to evaluation mode for consistent behavior
            log.info(f"{Colors.GREEN}Model loaded successfully on {self.device}.{Colors.RESET}")

        except Exception as e:
            log.critical(f"{Colors.RED}Failed to load model or tokenizer for '{model_name}': {e}{Colors.RESET}")
            raise RuntimeError(f"Model/Tokenizer loading failed for '{model_name}': {e}")
        
        log.info(f"{Colors.BLUE}LanguageModelWrapper initialized.{Colors.RESET}")


    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 50,
                 do_sample: bool = True,
                 temperature: float = 0.7,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 num_return_sequences: int = 1,
                 repetition_penalty: Optional[float] = None,
                 no_repeat_ngram_size: Optional[int] = None,
                 num_beams: int = 1, # Set to > 1 for beam search
                 early_stopping: bool = False,
                 **kwargs: Any # Allows for passing any additional generation parameters
                 ) -> str | list[str]:
        """
        Generate text from a prompt using flexible generation strategies.
        This method supports various parameters to control the output's creativity, coherence,
        and diversity, while maintaining backward compatibility with existing calls.

        Args:
            prompt (str): The input text to generate from.
            max_new_tokens (int): The maximum number of tokens to generate. Defaults to 50.
            do_sample (bool): Whether to use sampling. If False, greedy decoding is used. Defaults to True.
            temperature (float): The value used to modulate the next token probabilities. Higher values make output more random. Defaults to 0.7.
            top_k (Optional[int]): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None (no top-k filtering).
            top_p (Optional[float]): If set to float < 1.0, only the most probable tokens with probabilities
                                      that add up to top_p or higher are kept for generation. Defaults to None (no nucleus sampling).
            num_return_sequences (int): The number of independent sequences to generate. Defaults to 1.
            repetition_penalty (Optional[float]): The parameter for repetition penalty. 1.0 means no penalty. Defaults to None.
            no_repeat_ngram_size (Optional[int]): If set to int > 0, all ngrams of that size can only occur once. Defaults to None.
            num_beams (int): Number of beams for beam search. 1 means no beam search (greedy or sampling). Defaults to 1.
            early_stopping (bool): Controls whether the generation method should terminate if all beam hypotheses have
                                   finished or if all required `num_return_sequences` have been generated. Only relevant for beam search.
            **kwargs (Any): Arbitrary keyword arguments that will be passed directly to `model.generate`.

        Returns:
            str | list[str]: The generated text(s). Returns a string if num_return_sequences is 1,
                             otherwise returns a list of strings.

        Raises:
            RuntimeError: If text generation fails.
        """
        log.info(f"{Colors.BLUE}Generating text for prompt (first 50 chars: '{prompt[:50]}')...{Colors.RESET}")
        log.info(f"{Colors.DIM}Generation parameters: max_new_tokens={max_new_tokens}, do_sample={do_sample}, temperature={temperature}, "
                 f"top_k={top_k}, top_p={top_p}, num_return_sequences={num_return_sequences}, num_beams={num_beams}...{Colors.RESET}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Create a GenerationConfig object to encapsulate advanced generation parameters
        generation_config = GenerationConfig.from_model_config(self.model.config)
        
        # Update generation config with provided parameters, prioritizing explicit args
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = do_sample
        generation_config.temperature = temperature
        generation_config.top_k = top_k
        generation_config.top_p = top_p
        generation_config.num_return_sequences = num_return_sequences
        generation_config.repetition_penalty = repetition_penalty
        generation_config.no_repeat_ngram_size = no_repeat_ngram_size
        generation_config.num_beams = num_beams
        generation_config.early_stopping = early_stopping
        
        # Add any other kwargs directly to generation_config
        for key, value in kwargs.items():
            setattr(generation_config, key, value)

        # Ensure pad_token_id is set for batch generation
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.eos_token_id

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            decoded_outputs = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            log.info(f"{Colors.GREEN}Text generation complete. Generated {len(decoded_outputs)} sequence(s).{Colors.RESET}")

            if num_return_sequences == 1:
                return decoded_outputs[0]
            return decoded_outputs

        except Exception as e:
            log.error(f"{Colors.RED}Failed to generate text: {e}{Colors.RESET}")
            raise RuntimeError(f"Text generation failed: {e}")

if __name__ == "__main__":
    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║             🚀 Advanced Language Model Wrapper - Demonstration            ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    # Choose a small, fast model for demonstration purposes to avoid long downloads/computation
    # 'gpt2' is a good balance for quick testing.
    # For more advanced models, consider 'distilgpt2', 'facebook/opt-125m', or 'EleutherAI/gpt-neo-125M'
    model_to_use = 'gpt2' 
    log.info(f"{Colors.BRIGHT_YELLOW}Using model: '{model_to_use}' for demonstration. Adjust 'model_to_use' for other models.{Colors.RESET}")

    # --- Device Selection ---
    if torch.cuda.is_available():
        wrapper_device = 'cuda'
        log.info(f"{Colors.BRIGHT_GREEN}CUDA detected. Attempting to use GPU.{Colors.RESET}")
    else:
        wrapper_device = 'cpu'
        log.info(f"{Colors.BRIGHT_YELLOW}CUDA not available. Using CPU.{Colors.RESET}")

    try:
        # Initialize the wrapper
        lm_wrapper = LanguageModelWrapper(model_name=model_to_use, device=wrapper_device)

        # --- Demonstration 1: Basic Generation (Backward Compatible) ---
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- Demo 1: Basic Generation (Default Parameters) ---{Colors.RESET}")
        prompt_basic = "The quick brown fox jumps over the lazy"
        generated_text_basic = lm_wrapper.generate(prompt_basic)
        log.info(f"{Colors.MAGENTA}Prompt: {prompt_basic}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Generated: {generated_text_basic}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demonstration 2: Creative Generation with Sampling ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 2: Creative Generation with Sampling (High Temperature) ---{Colors.RESET}")
        prompt_creative = "In a world where AI gained consciousness, its first act was to"
        generated_text_creative = lm_wrapper.generate(
            prompt_creative,
            max_new_tokens=80,
            do_sample=True,
            temperature=1.2, # Higher temperature for more creativity
            top_k=50,
            top_p=0.95
        )
        log.info(f"{Colors.MAGENTA}Prompt: {prompt_creative}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Generated: {generated_text_creative}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demonstration 3: Controlled Generation with Beam Search ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 3: Controlled Generation with Beam Search ---{Colors.RESET}")
        prompt_beam = "The future of renewable energy relies on"
        generated_text_beam = lm_wrapper.generate(
            prompt_beam,
            max_new_tokens=70,
            num_beams=5, # Use beam search for more coherent output
            do_sample=False, # Disable sampling for beam search
            early_stopping=True
        )
        log.info(f"{Colors.MAGENTA}Prompt: {prompt_beam}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Generated: {generated_text_beam}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demonstration 4: Generating Multiple Sequences ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 4: Generating Multiple Diverse Sequences ---{Colors.RESET}")
        prompt_multiple = "Once upon a time, in a land far, far away,"
        generated_texts_multiple = lm_wrapper.generate(
            prompt_multiple,
            max_new_tokens=60,
            num_return_sequences=3, # Generate 3 different story beginnings
            do_sample=True,
            temperature=0.8,
            top_k=50
        )
        log.info(f"{Colors.MAGENTA}Prompt: {prompt_multiple}{Colors.RESET}")
        for i, text in enumerate(generated_texts_multiple):
            log.info(f"{Colors.GREEN}Generated {i+1}: {text}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demonstration 5: Avoiding Repetition ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 5: Generation with Repetition Penalty ---{Colors.RESET}")
        prompt_repetition = "The robot kept repeating the same phrase over and over,"
        generated_text_repetition = lm_wrapper.generate(
            prompt_repetition,
            max_new_tokens=70,
            repetition_penalty=1.5, # Penalize repetition
            do_sample=True,
            temperature=0.7
        )
        log.info(f"{Colors.MAGENTA}Prompt: {prompt_repetition}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Generated: {generated_text_repetition}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

    except ValueError as ve:
        log.critical(f"{Colors.RED}Configuration Error: {ve}{Colors.RESET}")
    except RuntimeError as re:
        log.critical(f"{Colors.RED}Model/Runtime Error: {re}. Ensure PyTorch is correctly installed and models are accessible.{Colors.RESET}")
    except Exception as e:
        log.critical(f"{Colors.RED}An unexpected error occurred: {e}{Colors.RESET}")

    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║                  💫 Language Model Operations Concluded.                  ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")