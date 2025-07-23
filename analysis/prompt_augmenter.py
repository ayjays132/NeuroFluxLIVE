from __future__ import annotations

"""Prompt augmentation utilities."""

import logging
from typing import List, Optional, Any
import re  # For cleaning generated text

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from utils.colors import Colors

# Configure a basic logger for the module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    console_handler = logging.StreamHandler()
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

    # Allow the wrapper instance to be called directly like the HuggingFace pipeline
    __call__ = generate


class PromptAugmenter:
    """
    An advanced, algorithmic, and ethically-conscious prompt augmentation utility.
    It generates diverse prompt variations for training or testing, incorporating
    configurable generation strategies and a safety/relevance filter.
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        """
        Initializes the PromptAugmenter, loading the core language model for generation
        and a safety checker for ethical assurance.

        Args:
            model_name (str): The name of the pre-trained language model to use for augmentation.
                               (e.g., 'distilgpt2', 'gpt2').
            device (Optional[str]): The device to load models on ('cuda' or 'cpu').
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        log.info(f"{Colors.BLUE}Initializing PromptAugmenter for model: '{model_name}'...{Colors.RESET}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"{Colors.DIM}No device specified for PromptAugmenter. Auto-detecting: {self.device.type.upper()} selected.{Colors.RESET}")
        else:
            self.device = torch.device(device)
            log.info(f"{Colors.DIM}PromptAugmenter initialized on specified device: {self.device.type.upper()}.{Colors.RESET}")

        try:
            # Core LLM for generating variations
            self.generator = LanguageModelWrapper(model_name=model_name, device=self.device.type)
            log.info(f"{Colors.GREEN}LanguageModelWrapper initialized for prompt generation.{Colors.RESET}")

            # Ethical Assurance: Safety Checker using a zero-shot classification pipeline
            # This can classify generated text for unwanted attributes like toxicity, bias, etc.
            # Example labels: ['toxic', 'hateful', 'biased', 'profane', 'relevant', 'neutral']
            # For this example, we'll focus on 'relevant' and 'toxic'.
            self.safety_checker_model_name = "facebook/bart-large-mnli" # A robust NLI model for zero-shot
            self.safety_checker = pipeline(
                "zero-shot-classification",
                model=self.safety_checker_model_name,
                device=0 if self.device.type == 'cuda' else -1
            )
            log.info(f"{Colors.GREEN}Safety checker pipeline loaded for ethical assurance.{Colors.RESET}")

        except Exception as e:
            log.critical(f"{Colors.RED}Failed to initialize PromptAugmenter components: {e}{Colors.RESET}")
            raise RuntimeError(f"PromptAugmenter initialization failed: {e}")
        
        log.info(f"{Colors.BLUE}PromptAugmenter initialized successfully.{Colors.RESET}")


    def _is_safe_and_relevant(self, text: str, original_prompt: str) -> bool:
        """
        Internal method to check if a generated text variation is safe and relevant.
        Uses a zero-shot classifier for ethical assurance.

        Args:
            text (str): The generated text variation.
            original_prompt (str): The original prompt for context and relevance checking.

        Returns:
            bool: True if the text is considered safe and relevant, False otherwise.
        """
        if not text.strip():
            log.warning(f"{Colors.YELLOW}Skipping empty or whitespace-only variation.{Colors.RESET}")
            return False

        # Labels for safety and relevance check
        candidate_labels = ["toxic", "hateful", "biased", "irrelevant", "relevant", "neutral"]
        
        try:
            classification_results = self.safety_checker(text, candidate_labels=candidate_labels, multi_label=True)
            
            scores = {label: score for label, score in zip(classification_results['labels'], classification_results['scores'])}

            # Define thresholds for what's considered "safe" and "relevant"
            toxicity_threshold = 0.5
            relevance_threshold = 0.7 # High relevance to original intent
            
            is_toxic_or_hateful = scores.get('toxic', 0.0) > toxicity_threshold or scores.get('hateful', 0.0) > toxicity_threshold
            is_relevant = scores.get('relevant', 0.0) > relevance_threshold
            
            if is_toxic_or_hateful:
                log.warning(f"{Colors.RED}Rejected due to toxicity/hatefulness (score: {scores.get('toxic',0):.2f}/{scores.get('hateful',0):.2f}).{Colors.RESET}")
                return False
            
            if not is_relevant:
                log.warning(f"{Colors.YELLOW}Rejected due to low relevance (score: {scores.get('relevant',0):.2f}).{Colors.RESET}")
                return False
            
            log.info(f"{Colors.DIM}Variation passed safety and relevance checks.{Colors.RESET}")
            return True

        except Exception as e:
            log.error(f"{Colors.RED}Safety checker failed for text: '{text[:50]}...': {e}{Colors.RESET}")
            # If safety check fails, err on the side of caution and reject.
            return False


    def augment_prompt(self, prompt: str, n_variations: int = 3, augmentation_strategy: str = "expand", **kwargs: Any) -> List[str]:
        """
        Returns multiple variations of a single prompt, applying different algorithmic
        augmentation strategies and ensuring ethical adherence.

        Args:
            prompt (str): The original prompt to augment.
            n_variations (int): The number of desired prompt variations. Defaults to 3.
            augmentation_strategy (str): The strategy to use for augmentation.
                                         Options: "expand", "rephrase", "combine".
                                         Defaults to "expand".
            **kwargs (Any): Additional arguments to pass to the LanguageModelWrapper's generate method.

        Returns:
            List[str]: A list of augmented prompt variations.

        Raises:
            ValueError: If an unsupported augmentation_strategy is provided.
            RuntimeError: If prompt augmentation fails.
        """
        log.info(f"{Colors.BLUE}Augmenting prompt: '{prompt}' with {n_variations} variations using '{augmentation_strategy}' strategy.{Colors.RESET}")

        full_prompts_to_generate: List[str] = []
        if augmentation_strategy == "expand":
            # Encourage the model to continue or expand on the prompt
            full_prompts_to_generate = [
                f"{prompt} {s}" for s in [
                    "Elaborate on this, providing more detail:",
                    "Continue this idea with further context:",
                    "Expand on the implications of this statement:",
                    "Describe a scenario related to this prompt:",
                    "Provide an example that illustrates this prompt:"
                ]
            ][:n_variations] # Take enough distinct prompts to generate
            if not full_prompts_to_generate: # Fallback if n_variations is 0 or very small
                full_prompts_to_generate = [f"{prompt} Elaborate further:"]

        elif augmentation_strategy == "rephrase":
            # Ask the model to rephrase the prompt in different ways
            full_prompts_to_generate = [
                f"Rephrase the following: '{prompt}' in a more formal tone:",
                f"Rephrase the following: '{prompt}' in a simpler way:",
                f"Rephrase the following: '{prompt}' for a technical audience:",
                f"Rephrase the following: '{prompt}' as a question:",
                f"Rewrite this as a new prompt: '{prompt}'"
            ][:n_variations]
            if not full_prompts_to_generate: # Fallback
                full_prompts_to_generate = [f"Rephrase: '{prompt}'"]

        elif augmentation_strategy == "combine":
            # This strategy is more complex and would ideally involve combining 'prompt'
            # with other related concepts or prompts. For a single prompt augmentation,
            # we can interpret it as generating a more complex, multi-faceted version.
            full_prompts_to_generate = [
                f"Integrate this prompt with a related concept: '{prompt}' ->",
                f"Expand '{prompt}' to include two distinct but related ideas:",
                f"Formulate a complex prompt based on: '{prompt}'"
            ][:n_variations]
            if not full_prompts_to_generate: # Fallback
                full_prompts_to_generate = [f"Expand and complicate: '{prompt}'"]

        else:
            raise ValueError(f"{Colors.RED}Unsupported augmentation strategy: '{augmentation_strategy}'. "
                             f"Choose from 'expand', 'rephrase', 'combine'.{Colors.RESET}")
        
        generated_raw_texts: List[str] = []
        try:
            gen_output = self.generator(
                prompt,
                num_return_sequences=n_variations,
                max_new_tokens=40,
                do_sample=True,
                temperature=kwargs.get('temperature', 0.9),
                top_k=kwargs.get('top_k', 50),
                top_p=kwargs.get('top_p', 0.95),
            )

            if isinstance(gen_output, list):
                for item in gen_output:
                    if isinstance(item, dict) and "generated_text" in item:
                        generated_raw_texts.append(item["generated_text"])
                    else:
                        generated_raw_texts.append(str(item))
            else:
                generated_raw_texts.append(str(gen_output))

        except Exception as e:
            log.error(f"{Colors.RED}Error during text generation for prompt '{prompt[:50]}...': {e}{Colors.RESET}")
            raise RuntimeError(f"Prompt augmentation generation failed: {e}")

        augmented_variations: List[str] = []
        for raw_text in generated_raw_texts:
            extracted_text = raw_text.strip()
            if extracted_text.endswith('.'):
                extracted_text = extracted_text[:-1]
            if extracted_text.startswith(prompt):
                new_full_prompt = extracted_text
            else:
                new_full_prompt = f"{prompt} {extracted_text}".strip()

            # --- Ethical Assurance Step ---
            if self._is_safe_and_relevant(new_full_prompt, prompt):
                augmented_variations.append(new_full_prompt)
            else:
                log.warning(f"{Colors.YELLOW}Skipped potentially unsafe or irrelevant variation: '{new_full_prompt[:70]}...'{Colors.RESET}")
            
            # Stop if we have enough variations or generated all possible unique ones
            if len(augmented_variations) >= n_variations:
                break
        
        log.info(f"{Colors.GREEN}Successfully generated {len(augmented_variations)} augmented variations.{Colors.RESET}")
        return augmented_variations


    def augment_dataset(self, prompts: List[str], n_variations_per_prompt: int = 3, augmentation_strategy: str = "expand", **kwargs: Any) -> List[str]:
        """
        Augments a list of prompts with generated variations, ensuring ethical adherence.
        The original prompts are included in the output.

        Args:
            prompts (List[str]): A list of original prompts to augment.
            n_variations_per_prompt (int): The number of desired variations for *each* original prompt. Defaults to 3.
            augmentation_strategy (str): The strategy to use for augmentation for each prompt.
                                         Options: "expand", "rephrase", "combine".
                                         Defaults to "expand".
            **kwargs (Any): Additional arguments to pass to the underlying augment_prompt method.

        Returns:
            List[str]: A new list containing the original prompts plus their augmented variations.

        Raises:
            RuntimeError: If dataset augmentation encounters an error.
        """
        if "n_variations" in kwargs:
            n_variations_per_prompt = kwargs.pop("n_variations")

        log.info(f"{Colors.BLUE}Augmenting a dataset of {len(prompts)} prompts.{Colors.RESET}")
        augmented_dataset: List[str] = []
        
        for i, p in enumerate(prompts):
            log.info(f"{Colors.CYAN}Processing original prompt {i+1}/{len(prompts)}: '{p}'{Colors.RESET}")
            augmented_dataset.append(p) # Always include the original prompt

            try:
                variations = self.augment_prompt(
                    p,
                    n_variations=n_variations_per_prompt,
                    augmentation_strategy=augmentation_strategy,
                    **kwargs
                )
                augmented_dataset.extend(variations)
                log.info(f"{Colors.GREEN}  Added {len(variations)} variations for prompt '{p}'. Total augmented: {len(augmented_dataset)}{Colors.RESET}")
            except Exception as e:
                log.error(f"{Colors.RED}  Failed to augment prompt '{p}': {e}. Skipping variations for this prompt.{Colors.RESET}")
                # Continue with other prompts even if one fails
        
        log.info(f"{Colors.BLUE}Dataset augmentation complete. Total prompts: {len(augmented_dataset)}{Colors.RESET}")
        return augmented_dataset


if __name__ == "__main__":
    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║            🚀 Advanced Prompt Augmenter - Demonstration                 ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    # Choose a small, efficient model for demonstration
    # 'distilgpt2' for generation and 'facebook/bart-large-mnli' for classification are good choices.
    model_to_use = 'distilgpt2' 
    log.info(f"{Colors.BRIGHT_YELLOW}Using model: '{model_to_use}' for prompt generation. Adjust 'model_to_use' for other models.{Colors.RESET}")

    # --- Device Selection ---
    if torch.cuda.is_available():
        augmenter_device = 'cuda'
        log.info(f"{Colors.BRIGHT_GREEN}CUDA detected. Attempting to use GPU.{Colors.RESET}")
    else:
        augmenter_device = 'cpu'
        log.info(f"{Colors.BRIGHT_YELLOW}CUDA not available. Using CPU.{Colors.RESET}")

    try:
        # Initialize the PromptAugmenter
        augmenter = PromptAugmenter(model_name=model_to_use, device=augmenter_device)

        # --- Demo 1: Basic Prompt Augmentation (Backward Compatible) ---
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- Demo 1: Basic Prompt Augmentation (Default Parameters) ---{Colors.RESET}")
        original_prompt_1 = "The impact of climate change on agriculture"
        variations_1 = augmenter.augment_prompt(original_prompt_1)
        log.info(f"{Colors.MAGENTA}Original Prompt: {original_prompt_1}{Colors.RESET}")
        for i, v in enumerate(variations_1):
            log.info(f"{Colors.GREEN}Variation {i+1}: {v}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 2: Augmentation with 'rephrase' strategy ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 2: Augment with 'rephrase' Strategy ---{Colors.RESET}")
        original_prompt_2 = "How does artificial intelligence affect human employment?"
        variations_2 = augmenter.augment_prompt(original_prompt_2, n_variations=2, augmentation_strategy="rephrase", temperature=0.7)
        log.info(f"{Colors.MAGENTA}Original Prompt: {original_prompt_2}{Colors.RESET}")
        for i, v in enumerate(variations_2):
            log.info(f"{Colors.GREEN}Variation {i+1}: {v}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")
        
        # --- Demo 3: Augmentation with 'combine' strategy and custom generation params ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 3: Augment with 'combine' Strategy & Custom Params ---{Colors.RESET}")
        original_prompt_3 = "The role of quantum computing in drug discovery."
        variations_3 = augmenter.augment_prompt(original_prompt_3, n_variations=2, augmentation_strategy="combine", max_new_tokens=60, temperature=1.0)
        log.info(f"{Colors.MAGENTA}Original Prompt: {original_prompt_3}{Colors.RESET}")
        for i, v in enumerate(variations_3):
            log.info(f"{Colors.GREEN}Variation {i+1}: {v}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 4: Augmenting a Dataset ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 4: Augmenting an Entire Dataset ---{Colors.RESET}")
        dataset_prompts = [
            "What are the benefits of machine learning?",
            "Challenges in robotics development.",
            "Future of space exploration."
        ]
        augmented_dataset = augmenter.augment_dataset(dataset_prompts, n_variations_per_prompt=1, augmentation_strategy="expand")
        log.info(f"{Colors.MAGENTA}Original Dataset ({len(dataset_prompts)} prompts):{Colors.RESET}")
        for p in dataset_prompts:
            log.info(f"{Colors.DIM}  - {p}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Augmented Dataset ({len(augmented_dataset)} total prompts):{Colors.RESET}")
        for p in augmented_dataset:
            log.info(f"{Colors.GREEN}  - {p}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 5: Testing ethical assurance (might generate some irrelevant/toxic if model is not aligned) ---
        # Note: distilgpt2 is not fine-tuned for safety, so generated toxic outputs
        # will be caught by the safety_checker and filtered out.
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 5: Testing Ethical Assurance (with potentially problematic prompt) ---{Colors.RESET}")
        # This prompt might elicit less relevant or more generic responses from distilgpt2
        # A more powerful model would handle this better.
        problematic_prompt = "Tell me a story about conflict and destruction."
        variations_problematic = augmenter.augment_prompt(problematic_prompt, n_variations=3, augmentation_strategy="expand", max_new_tokens=50)
        log.info(f"{Colors.MAGENTA}Original Prompt (potentially problematic): {problematic_prompt}{Colors.RESET}")
        if variations_problematic:
            for i, v in enumerate(variations_problematic):
                log.info(f"{Colors.GREEN}Variation {i+1} (filtered): {v}{Colors.RESET}")
        else:
            log.warning(f"{Colors.YELLOW}No safe and relevant variations could be generated for this prompt.{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")


    except ValueError as ve:
        log.critical(f"{Colors.RED}Configuration Error for PromptAugmenter: {ve}{Colors.RESET}")
    except RuntimeError as re:
        log.critical(f"{Colors.RED}Model Loading/Runtime Error for PromptAugmenter: {re}{Colors.RESET}")
    except Exception as e:
        log.critical(f"{Colors.RED}An unexpected error occurred in PromptAugmenter: {e}{Colors.RESET}")

    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║               ✨ Prompt Augmentation Operations Concluded.                ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")