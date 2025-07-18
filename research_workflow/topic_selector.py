from __future__ import annotations

"""Model wrapper utilities for HuggingFace models."""

import logging
from typing import Optional, Any, List
import re  # For parsing generated text

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


class TopicSelector:
    """
    A robust and advanced class to select and validate research topics and questions.
    It leverages the LanguageModelWrapper for intelligent suggestion and validation
    using generative AI capabilities.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initializes the TopicSelector with a specified device and loads required LLMs
        through the LanguageModelWrapper.

        Args:
            device (Optional[str]): The device to use for computations ('cuda' or 'cpu').
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        log.info(f"{Colors.BLUE}Initializing TopicSelector...{Colors.RESET}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"{Colors.DIM}No device specified for TopicSelector. Auto-detecting: {self.device.type.upper()} selected.{Colors.RESET}")
        else:
            self.device = torch.device(device)
            log.info(f"{Colors.DIM}TopicSelector initialized on specified device: {self.device.type.upper()}.{Colors.RESET}")

        # Using a small, efficient pre-trained language model for topic generation
        # 'distilgpt2' is a good balance for quick testing.
        # For more advanced models in a real scenario, consider 'facebook/opt-125m', 'EleutherAI/gpt-neo-125M', or larger models.
        self.generator_model_name = "distilgpt2"
        self.validator_model_name = "distilgpt2" # Can use the same model for both or a different one

        try:
            self.generator = LanguageModelWrapper(model_name=self.generator_model_name, device=self.device.type)
            log.info(f"{Colors.GREEN}Generator LanguageModelWrapper loaded for topic suggestion.{Colors.RESET}")

            # For question validation, we'll also use LanguageModelWrapper
            self.validator = LanguageModelWrapper(model_name=self.validator_model_name, device=self.device.type)
            log.info(f"{Colors.GREEN}Validator LanguageModelWrapper loaded for question validation.{Colors.RESET}")

            # Also keep the sentiment analysis pipeline for a simpler, quick check and to
            # ensure some form of the original logic remains, though the LLM-based validation
            # is more robust.
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english", 
                device=0 if self.device.type == 'cuda' else -1
            )
            log.info(f"{Colors.GREEN}Sentiment analysis pipeline loaded for secondary validation.{Colors.RESET}")

        except Exception as e:
            log.critical(f"{Colors.RED}Failed to initialize LanguageModelWrapper for TopicSelector: {e}{Colors.RESET}")
            raise RuntimeError(f"TopicSelector initialization failed: {e}")
        
        log.info(f"{Colors.BLUE}TopicSelector initialized successfully.{Colors.RESET}")


    def suggest_topic(self, research_area: str, num_suggestions: int = 1, **kwargs: Any) -> str | List[str]:
        """
        Suggests one or more highly specific and novel research topics based on the
        input research area using the generative LLM.

        Args:
            research_area (str): The broad area of research.
            num_suggestions (int): The number of distinct topics to suggest. Defaults to 1.
            **kwargs (Any): Additional arguments to pass to the underlying LanguageModelWrapper's generate method.

        Returns:
            str | List[str]: A single suggested research topic (str) if num_suggestions is 1,
                             otherwise a list of suggested research topics (List[str]).

        Raises:
            RuntimeError: If topic generation fails.
        """
        log.info(f"{Colors.BLUE}Requesting {num_suggestions} research topic(s) for area: '{research_area}'...{Colors.RESET}")
        
        prompt = (f"Generate a highly specific, novel, and academically suitable research topic "
                  f"in the field of {research_area}. Avoid common knowledge topics. "
                  f"Focus on emerging trends or interdisciplinary approaches. "
                  f"The topic should be concise and impactful. "
                  f"Research Topic:")
        
        try:
            # Generate text using the LanguageModelWrapper
            generated_texts = self.generator.generate(
                prompt,
                num_return_sequences=num_suggestions,
                max_new_tokens=70, # Increased max_new_tokens for more detailed suggestions
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                **kwargs # Pass through any additional generation arguments
            )

            # Ensure generated_texts is a list for consistent processing
            if isinstance(generated_texts, str):
                generated_texts = [generated_texts]

            suggested_topics = []
            for text in generated_texts:
                # Extract the generated topic by carefully removing the prompt
                # The LLM might sometimes include parts of the prompt or preamble
                topic_raw = text[len(prompt):].strip()
                
                # Clean up: remove trailing punctuation or incomplete sentences
                # Only keep the first logical sentence if multiple are generated
                first_sentence_match = re.match(r"([^.?!]*[.?!])", topic_raw)
                if first_sentence_match:
                    cleaned_topic = first_sentence_match.group(1).strip()
                else:
                    cleaned_topic = topic_raw.split('\n')[0].strip() # Take first line
                
                # Further refine by removing any remaining prompt fragments or conversational filler
                cleaned_topic = re.sub(r"^(Here is|Suggested Topic:|Topic:)\s*", "", cleaned_topic, flags=re.IGNORECASE).strip()
                if cleaned_topic.endswith('.'): # Remove trailing period if present
                    cleaned_topic = cleaned_topic[:-1]
                
                suggested_topics.append(cleaned_topic)
                log.info(f"{Colors.GREEN}  Generated Topic: {cleaned_topic}{Colors.RESET}")


            if num_suggestions == 1:
                return suggested_topics[0] if suggested_topics else ""
            return suggested_topics if suggested_topics else [""] * num_suggestions

        except Exception as e:
            log.error(f"{Colors.RED}Failed to suggest topic for '{research_area}': {e}{Colors.RESET}")
            raise RuntimeError(f"Topic suggestion failed: {e}")


    def validate_question(self, question: str, **kwargs: Any) -> bool:
        """
        Validates a research question for clarity, focus, feasibility, and academic suitability
        using a generative LLM. It provides detailed feedback and a clear pass/fail status.

        Args:
            question (str): The research question to validate.
            **kwargs (Any): Additional arguments to pass to the underlying LanguageModelWrapper's generate method for validation.

        Returns:
            bool: True if the question is considered valid, False otherwise.

        Raises:
            RuntimeError: If question validation fails.
        """
        log.info(f"{Colors.BLUE}Validating research question: '{question}'...{Colors.RESET}")

        # --- LLM-based Validation (Primary, more robust) ---
        validation_prompt = (
            f"Evaluate the following research question for clarity, focus, feasibility, and academic suitability. "
            f"Provide a brief feedback and then state 'VALID: YES' or 'VALID: NO'.\n\n"
            f"Research Question: \"{question}\"\n\n"
            f"Evaluation:"
        )

        try:
            validation_response = self.validator.generate(
                validation_prompt,
                max_new_tokens=100, # More tokens for detailed feedback
                do_sample=False,    # Aim for deterministic evaluation
                temperature=0.1,
                num_return_sequences=1,
                **kwargs
            )
            log.info(f"{Colors.DIM}LLM Validation Response: {validation_response}{Colors.RESET}")

            # Parse the LLM's response
            is_valid_llm = False
            feedback = "No specific feedback from LLM."
            
            # Extract feedback
            feedback_match = re.search(r"Evaluation:\s*(.*?)(VALID: (?:YES|NO))", validation_response, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
                if feedback.endswith('.'):
                    feedback = feedback[:-1]

            # Check for the explicit VALID: YES/NO tag
            valid_tag_match = re.search(r"VALID: (YES|NO)", validation_response)
            if valid_tag_match:
                if valid_tag_match.group(1) == 'YES':
                    is_valid_llm = True
            
            log.info(f"{Colors.YELLOW}LLM Feedback: {feedback}{Colors.RESET}")
            log.info(f"{Colors.YELLOW}LLM Valid Status: {is_valid_llm}{Colors.RESET}")

            # --- Sentiment-based Validation (Secondary, original approach re-integrated) ---
            sentiment_result = self.sentiment_pipeline(question)
            is_positive_sentiment = False
            if sentiment_result and sentiment_result[0]['label'] == 'POSITIVE' and sentiment_result[0]['score'] > 0.8:
                is_positive_sentiment = True
            
            log.info(f"{Colors.DIM}Sentiment Analysis Result: {sentiment_result[0]['label']} (Score: {sentiment_result[0]['score']:.2f}){Colors.RESET}")
            log.info(f"{Colors.DIM}Sentiment indicates positive: {is_positive_sentiment}{Colors.RESET}")

            # --- Combined Validation Logic ---
            # A question is considered valid if both LLM says YES and it passes basic structural/sentiment checks
            final_is_valid = is_valid_llm and is_positive_sentiment and ('?' in question and len(question.split()) > 5)

            if final_is_valid:
                log.info(f"{Colors.GREEN}{Colors.BOLD}Question '{question}' is VALID.{Colors.RESET}")
            else:
                log.info(f"{Colors.RED}{Colors.BOLD}Question '{question}' is INVALID.{Colors.RESET}")
                if not is_valid_llm:
                    log.info(f"{Colors.RED}Reason: LLM validation explicitly marked it as 'NO'.{Colors.RESET}")
                elif not is_positive_sentiment:
                    log.info(f"{Colors.RED}Reason: Sentiment analysis was not sufficiently positive.{Colors.RESET}")
                elif '?' not in question:
                    log.info(f"{Colors.RED}Reason: Missing a question mark.{Colors.RESET}")
                elif len(question.split()) <= 5:
                    log.info(f"{Colors.RED}Reason: Question is too short.{Colors.RESET}")

            return final_is_valid

        except Exception as e:
            log.error(f"{Colors.RED}Failed to validate question '{question}': {e}{Colors.RESET}")
            raise RuntimeError(f"Question validation failed: {e}")


if __name__ == "__main__":
    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║             ✨ Advanced Topic Selector - Demonstration                 ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    # --- Device Selection ---
    if torch.cuda.is_available():
        device_to_use = 'cuda'
        log.info(f"{Colors.BRIGHT_GREEN}CUDA is available! Using GPU for TopicSelector.{Colors.RESET}")
    else:
        device_to_use = 'cpu'
        log.info(f"{Colors.BRIGHT_YELLOW}CUDA not available. Using CPU for TopicSelector.{Colors.RESET}")

    try:
        selector = TopicSelector(device=device_to_use)

        # --- Demo 1: Basic Topic Suggestion ---
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- Demo 1: Basic Topic Suggestion ---{Colors.RESET}")
        area = "renewable energy"
        suggested_topic_1 = selector.suggest_topic(area)
        log.info(f"{Colors.MAGENTA}Research Area: {area}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Suggested Topic: {suggested_topic_1}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 2: Multiple Topic Suggestions ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 2: Multiple Topic Suggestions ---{Colors.RESET}")
        area_medicine = "personalized medicine"
        suggested_topics_multiple = selector.suggest_topic(area_medicine, num_suggestions=3, temperature=0.9, top_k=50)
        log.info(f"{Colors.MAGENTA}Research Area: {area_medicine}{Colors.RESET}")
        for i, topic in enumerate(suggested_topics_multiple):
            log.info(f"{Colors.GREEN}Suggested Topic {i+1}: {topic}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 3: Validating a Good Question ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 3: Validating a Good Question ---{Colors.RESET}")
        question1 = "How can blockchain technology enhance data security in federated learning for medical diagnostics?"
        is_valid1 = selector.validate_question(question1)
        log.info(f"{Colors.MAGENTA}Question: '{question1}'{Colors.RESET}")
        log.info(f"{Colors.GREEN}Is Valid: {is_valid1}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 4: Validating a Poor Question (Too Short) ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 4: Validating a Poor Question (Too Short) ---{Colors.RESET}")
        question2 = "AI good?"
        is_valid2 = selector.validate_question(question2)
        log.info(f"{Colors.MAGENTA}Question: '{question2}'{Colors.RESET}")
        log.info(f"{Colors.RED}Is Valid: {is_valid2}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 5: Validating a Poor Question (Unclear) ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 5: Validating a Poor Question (Unclear/Broad) ---{Colors.RESET}")
        question3 = "What about global warming?"
        is_valid3 = selector.validate_question(question3)
        log.info(f"{Colors.MAGENTA}Question: '{question3}'{Colors.RESET}")
        log.info(f"{Colors.RED}Is Valid: {is_valid3}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

        # --- Demo 6: Validating a Question without a Question Mark ---
        log.info(f"{Colors.CYAN}{Colors.BOLD}--- Demo 6: Validating a Question without a Question Mark ---{Colors.RESET}")
        question4 = "This is a statement about climate change impact on polar bears"
        is_valid4 = selector.validate_question(question4)
        log.info(f"{Colors.MAGENTA}Question: '{question4}'{Colors.RESET}")
        log.info(f"{Colors.RED}Is Valid: {is_valid4}{Colors.RESET}")
        log.info(f"{Colors.BRIGHT_CYAN}-------------------------------------------------------{Colors.RESET}\n")

    except ValueError as ve:
        log.critical(f"{Colors.RED}Configuration Error for TopicSelector: {ve}{Colors.RESET}")
    except RuntimeError as re:
        log.critical(f"{Colors.RED}Model Loading/Runtime Error for TopicSelector: {re}{Colors.RESET}")
    except Exception as e:
        log.critical(f"{Colors.RED}An unexpected error occurred in TopicSelector: {e}{Colors.RESET}")

    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║               🎉 Topic Selection Operations Concluded.                    ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")