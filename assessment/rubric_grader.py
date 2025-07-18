import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline, set_seed
import logging
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Tuple
import yaml # Added for rubric loading example, ensure you have pyyaml installed: pip install pyyaml
import re # For advanced text processing in feedback generation

# --- ANSI Escape Codes for Colors and Styles (for futuristic CMD output) ---
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

# Set seed for reproducibility in generative models
set_seed(42)

class RubricGrader:
    """
    An advanced, automated rubric-based grader for textual research proposals/reports.
    Leverages pre-trained transformer models for robust semantic similarity calculation
    and generative AI for nuanced, empathetic, and actionable feedback.

    Designed for high reliability and production environments, with comprehensive
    error handling, detailed logging, and optimization features like rubric embedding caching.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the RubricGrader with a specified device and loads NLP models.
        The device selection ensures optimal performance, leveraging GPU if available.

        Args:
            device (str): The device to use for computations ('cuda' or 'cpu').

        Raises:
            ValueError: If an unsupported device string is provided.
            RuntimeError: If 'cuda' is specified but not available, or model loading fails.
        """
        log.info(f"{Colors.BLUE}Initializing RubricGrader...{Colors.RESET}")
        
        # --- Device Setup ---
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device(device)
                log.info(f"{Colors.GREEN}{Colors.BOLD}RubricGrader initialized on GPU: {self.device}{Colors.RESET}")
            else:
                log.warning(f"{Colors.YELLOW}CUDA is not available. Falling back to CPU for RubricGrader.{Colors.RESET}")
                self.device = torch.device('cpu')
                log.info(f"{Colors.BLUE}RubricGrader initialized on CPU (CUDA unavailable).{Colors.RESET}")
        elif device == 'cpu':
            self.device = torch.device('cpu')
            log.info(f"{Colors.BLUE}RubricGrader initialized on CPU.{Colors.RESET}")
        else:
            raise ValueError(f"{Colors.RED}Unsupported device: '{device}'. Please choose 'cpu' or 'cuda'.{Colors.RESET}")

        # Load a pre-trained model for sentence embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

        # Initialize a text generation pipeline for feedback
        self.feedback_generator_pipeline = pipeline("text-generation", model="distilgpt2", device=0 if self.device.type == 'cuda' else -1)
        # --- Model Loading ---
        try:
            log.info(f"{Colors.CYAN}Loading sentence embedding model (all-MiniLM-L6-v2)...{Colors.RESET}")
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
            log.info(f"{Colors.GREEN}Sentence embedding model loaded on {self.device}.{Colors.RESET}")
            
            log.info(f"{Colors.CYAN}Loading feedback generation model (ayjays132/NeuroReasoner-1-NR-1)...{Colors.RESET}")
            pipeline_device_id = 0 if self.device.type == 'cuda' else -1 # 0 for first GPU, -1 for CPU
            self.feedback_generator_pipeline = pipeline(
                "text-generation", 
                model="ayjays132/NeuroReasoner-1-NR-1", 
                tokenizer=self.tokenizer, # Ensure tokenizer is consistent
                device=pipeline_device_id
            )
            log.info(f"{Colors.GREEN}Feedback generation pipeline loaded on device: {self.feedback_generator_pipeline.device}.{Colors.RESET}")

        except Exception as e:
            log.critical(f"{Colors.RED}Failed to load one or more NLP models: {e}{Colors.RESET}")
            raise RuntimeError(f"Model loading failed for RubricGrader: {e}")

        # Cache for rubric embeddings to avoid recomputing for every submission
        self._rubric_embeddings_cache: Dict[str, torch.Tensor] = {}
        self._current_rubric_hash: Optional[int] = None # To track if rubric has changed

    def _hash_rubric(self, rubric: Dict[str, Any]) -> int:
        """Generates a simple hash for the rubric to check for changes."""
        # This is a simple hash. For production, consider a more robust hashing if rubrics are very dynamic.
        return hash(str(sorted(rubric.items())))

    def _load_rubric_embeddings(self, rubric: Dict[str, Any]):
        """
        Computes and caches embeddings for all expected content in the rubric.
        This optimizes performance by avoiding redundant computations when grading multiple submissions
        against the same rubric.
        """
        rubric_hash = self._hash_rubric(rubric)
        if rubric_hash == self._current_rubric_hash:
            log.info(f"{Colors.DIM}Rubric embeddings already cached and up-to-date. Skipping re-computation.{Colors.RESET}")
            return
        
        log.info(f"{Colors.BLUE}Loading and caching rubric embeddings...{Colors.RESET}")
        new_cache: Dict[str, torch.Tensor] = {}
        for criterion_name, criterion_details in rubric.items():
            if not isinstance(criterion_details, dict):
                log.warning(f"{Colors.YELLOW}Skipping non-criterion item '{criterion_name}' in rubric during embedding cache (expected dict, got {type(criterion_details).__name__}).{Colors.RESET}")
                continue
            expected_content = criterion_details.get('expected_content', '')
            if expected_content:
                try:
                    new_cache[criterion_name] = self._get_sentence_embedding(expected_content)
                except Exception as e:
                    log.error(f"{Colors.RED}Error embedding expected content for '{criterion_name}': {e}. This criterion may not be graded accurately.{Colors.RESET}")
            else:
                log.warning(f"{Colors.YELLOW}Criterion '{criterion_name}' has no 'expected_content'. No embedding will be cached.{Colors.RESET}")
        
        self._rubric_embeddings_cache = new_cache
        self._current_rubric_hash = rubric_hash
        log.info(f"{Colors.GREEN}Rubric embeddings cached successfully for {len(self._rubric_embeddings_cache)} criteria.{Colors.RESET}")


    def _get_sentence_embedding(self, text: str) -> torch.Tensor:
        """
        Generates a sentence embedding for the given text using the pre-loaded model.
        Handles empty or invalid text gracefully by returning a zero tensor.

        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: The sentence embedding.

        Raises:
            ValueError: If input text is empty or invalid (already handled by returning zero tensor).
        """
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        if not isinstance(text, str) or not text.strip():
            log.warning(f"{Colors.YELLOW}Received empty or invalid text for embedding. Returning zero tensor.{Colors.RESET}")
            return torch.zeros(1, self.model.config.hidden_size, device=self.device) # Use model's hidden_size dynamically

        encoded_input = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=self.tokenizer.model_max_length # Ensure max_length is respected
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Mean pooling to get sentence embedding
        sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Applies mean pooling to get sentence embedding from token embeddings."""
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Sum of token embeddings, masked by attention, divided by the number of active tokens
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _grade_criterion(self, submission_text: str, criterion_name: str, criterion_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grades a single criterion for a submission, calculating similarity, score, and generating feedback.

        Args:
            submission_text (str): The text of the student's submission.
            rubric (dict): A dictionary representing the rubric, e.g.,
                           {
                               "criterion1": {"expected_content": "...", "max_score": 10},
                               "criterion2": {"expected_content": "...", "max_score": 15}
                           }
            submission_text (str): The text content of the submission.
            criterion_name (str): The name of the criterion.
            criterion_details (Dict[str, Any]): Details of the criterion including 'expected_content' and 'max_score'.

        Returns:
            Dict[str, Any]: Grading results for the criterion (score, similarity, feedback).
        """
        log.info(f"{Colors.BLUE}  Processing criterion: '{criterion_name}'{Colors.RESET}")
        expected_content = criterion_details.get('expected_content', '')
        max_score = criterion_details.get('max_score', 0)

        if not expected_content:
            log.warning(f"{Colors.YELLOW}    Criterion '{criterion_name}' has no expected content. Skipping similarity calculation.{Colors.RESET}")
            return {
                "score": 0.0,
                "max_score": max_score,
                "similarity": 0.0,
                "feedback": "No expected content defined for this criterion. Score is 0."
            }
        
        if max_score <= 0:
            log.warning(f"{Colors.YELLOW}    Criterion '{criterion_name}' has a max_score of 0 or less. Score will be 0 regardless of similarity.{Colors.RESET}")

        # Retrieve expected content embedding from cache
        expected_embedding = self._rubric_embeddings_cache.get(criterion_name)
        if expected_embedding is None:
            # Fallback if not in cache (e.g., if rubric changed dynamically or error during cache loading)
            log.warning(f"{Colors.YELLOW}    Expected embedding for '{criterion_name}' not found in cache. Computing on-the-fly.{Colors.RESET}")
            expected_embedding = self._get_sentence_embedding(expected_content)

        submission_embedding = self._get_sentence_embedding(submission_text)

        # Ensure embeddings are on CPU for numpy. If they are already on CPU, this is a no-op.
        # Cosine similarity from sklearn expects numpy arrays.
        similarity = cosine_similarity(submission_embedding.cpu().numpy(), expected_embedding.cpu().numpy())[0][0]
        
        # Clamp similarity between 0 and 1, as it can sometimes be slightly outside due to floating point precision
        similarity = max(0.0, min(1.0, similarity))

        score = similarity * max_score
        
        # Generate feedback using the generative model
        feedback = self.generate_feedback(submission_text, expected_content, score, max_score)
        
        return {
            "score": score,
            "max_score": max_score,
            "similarity": similarity,
            "feedback": feedback
        }

    def _calculate_overall_summary(self, grades: Dict[str, Any], rubric: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the overall total score, percentage, and generates a holistic summary
        and actionable recommendations for the entire submission.

        Args:
            grades (Dict[str, Any]): The detailed grading results for each criterion.
            rubric (Dict[str, Any]): The rubric used for grading.

        Returns:
            Dict[str, Any]: A dictionary containing 'total_score', 'max_total_score',
                            'percentage', and 'overall_feedback'.
        """
        log.info(f"{Colors.BLUE}Calculating overall submission summary...{Colors.RESET}")
        total_score = sum(result.get('score', 0.0) for result in grades.values())
        max_total_score = sum(criterion.get('max_score', 0) for criterion in rubric.values() if isinstance(criterion, dict))

        percentage = (total_score / max_total_score * 100) if max_total_score > 0 else 0.0

        # Refactored to avoid nested f-string complexity for clarity and robustness
        criterion_feedback_snippets = []
        for c, g in grades.items():
            if c != "overall_summary": # Exclude overall_summary from this detailed list
                score_val = g.get("score", 0)
                max_score_val = g.get("max_score", 0)
                feedback_val = g.get("feedback", "")
                criterion_feedback_snippets.append(f"{c}: Score {score_val:.2f}/{max_score_val:.2f}, Feedback: {feedback_val}")
        
        detailed_feedback_string = " ".join(criterion_feedback_snippets)
        
        summary_prompt = (
            f"Based on the following detailed criterion scores and feedback for a submission:\n"
            f"{detailed_feedback_string}\n\n" # Using the pre-formatted string here
            f"The submission achieved a total score of {total_score:.2f} out of {max_total_score:.2f} ({percentage:.2f}%).\n"
            f"As an expert academic reviewer, provide a concise (2-3 sentences), empathetic, and actionable overall summary feedback. "
            f"Highlight overall strengths and areas for significant improvement. Overall Feedback:"
        )

        overall_feedback = "Could not generate overall feedback."
        try:
            generated_summary = self.feedback_generator_pipeline(
                summary_prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8, # Slightly higher temperature for more diverse summary
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True # Added truncation to handle long prompts and prevent CUDA errors
            )[0]['generated_text']

            overall_feedback = generated_summary[len(summary_prompt):].strip()
            # Clean up generated text (similar to generate_feedback)
            if '\n' in overall_feedback:
                overall_feedback = overall_feedback.split('\n')[0]
            if len(overall_feedback) > 10 and overall_feedback[-1] not in ['.', '!', '?']:
                last_full_stop = overall_feedback.rfind('.')
                if last_full_stop != -1 and last_full_stop > len(overall_feedback) * 0.7: # Only cut if near the end
                    overall_feedback = overall_feedback[:last_full_stop + 1]
                else: # If still incomplete, and no full stop nearby, cut at a reasonable length
                    overall_feedback = overall_feedback[:min(len(overall_feedback), 100)].strip() + "..."
            
            # Remove any trailing newlines or extra spaces
            overall_feedback = overall_feedback.strip()

        except Exception as e:
            log.error(f"{Colors.RED}Failed to generate overall summary feedback: {e}{Colors.RESET}")
            overall_feedback = f"Automated overall feedback generation failed due to an internal error: {e}"

        log.info(f"{Colors.GREEN}Overall summary calculated: {percentage:.2f}% ({total_score:.2f}/{max_total_score:.2f}){Colors.RESET}")
        
        return {
            "total_score": total_score,
            "max_total_score": max_total_score,
            "percentage": percentage,
            "overall_feedback": overall_feedback
        }


    def grade_submission(self, submission_text: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grades a single text submission against the provided rubric, including an overall summary.

        Args:
            submission_text (str): The text content of the submission to be graded.
            rubric (Dict[str, Any]): A dictionary representing the rubric, where keys are criterion names
                                     and values are dictionaries containing 'expected_content' and 'max_score'.

        Returns:
            Dict[str, Any]: A comprehensive dictionary containing criterion-wise grading results and an
                            'overall_summary' with total score, percentage, and holistic feedback.
        
        Raises:
            ValueError: If submission_text is empty or not a string, or if the rubric is invalid.
        """
        if not isinstance(submission_text, str) or not submission_text.strip():
            log.error(f"{Colors.RED}Invalid submission text provided: must be a non-empty string.{Colors.RESET}")
            raise ValueError("Submission text cannot be empty or invalid.")
        
        if not isinstance(rubric, dict) or not rubric:
            log.error(f"{Colors.RED}Invalid rubric provided: must be a non-empty dictionary.{Colors.RESET}")
            raise ValueError("Rubric cannot be empty or invalid.")

        # Ensure rubric embeddings are loaded/cached for this rubric
        self._load_rubric_embeddings(rubric)

        results: Dict[str, Any] = {}
        log.info(f"{Colors.BLUE}Grading submission (first 100 chars: '{submission_text[:100]}...') against rubric...{Colors.RESET}")

        for criterion_name, criterion_details in rubric.items():
            if not isinstance(criterion_details, dict):
                log.warning(f"{Colors.YELLOW}Skipping non-criterion item '{criterion_name}' in rubric (expected dict, got {type(criterion_details).__name__}).{Colors.RESET}")
                continue

            try:
                criterion_result = self._grade_criterion(submission_text, criterion_name, criterion_details)
                results[criterion_name] = criterion_result
            except Exception as e:
                log.error(f"{Colors.RED}Error grading criterion '{criterion_name}': {e}. Appending error result.{Colors.RESET}")
                results[criterion_name] = {
                    "error": str(e), 
                    "score": 0.0, 
                    "max_score": criterion_details.get('max_score', 0), 
                    "similarity": 0.0, 
                    "feedback": f"Error during grading: {e}"
                }
        
        overall_summary = self._calculate_overall_summary(results, rubric)
        results["overall_summary"] = overall_summary # Add overall summary to the results

        log.info(f"{Colors.GREEN}Submission grading complete with overall summary.{Colors.RESET}")
        return results

    def grade_batch(self, submissions: List[str], rubric: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Grades a batch of text submissions against the provided rubric.

        Args:
            submissions (List[str]): A list of text submissions to be graded.
            rubric (Dict[str, Any]): A dictionary representing the rubric.

        Returns:
            List[Dict[str, Any]]: A list of grading results, one comprehensive dictionary per submission.
        
        Raises:
            ValueError: If the submissions list is empty or invalid, or if the rubric is invalid.
        """
        if not isinstance(submissions, list) or not submissions:
            log.error(f"{Colors.RED}Invalid submissions list provided: must be a non-empty list of strings.{Colors.RESET}")
            raise ValueError("Submissions list cannot be empty or invalid.")
        
        if not isinstance(rubric, dict) or not rubric:
            log.error(f"{Colors.RED}Invalid rubric provided: must be a non-empty dictionary.{Colors.RESET}")
            raise ValueError("Rubric cannot be empty or invalid.")

        # Ensure rubric embeddings are loaded/cached once for the whole batch
        self._load_rubric_embeddings(rubric)

        batch_results: List[Dict[str, Any]] = []
        log.info(f"{Colors.BLUE}Starting batch grading for {len(submissions)} submissions.{Colors.RESET}")

        for i, submission_text in enumerate(submissions):
            log.info(f"{Colors.CYAN}--- Grading Submission {i+1}/{len(submissions)} ---{Colors.RESET}")
            try:
                # Use the grade_submission method for each submission, which now includes overall summary
                submission_grades = self.grade_submission(submission_text, rubric)
                batch_results.append(submission_grades)
                log.info(f"{Colors.GREEN}Successfully graded submission {i+1}.{Colors.RESET}")
            except Exception as e:
                log.error(f"{Colors.RED}Error grading submission {i+1}: {e}. Appending error result for this submission.{Colors.RESET}")
                batch_results.append({"error": str(e), "overall_summary": {"total_score": 0.0, "percentage": 0.0, "overall_feedback": f"Batch grading error for this submission: {e}"}})

        log.info(f"{Colors.GREEN}Batch grading complete.{Colors.RESET}")
        return batch_results

    def generate_feedback(self, submission_text: str, expected_content: str, score: float, max_score: float) -> str:
        """
        Generates actionable, empathetic, and nuanced feedback based on the criterion's score
        using a generative model. Incorporates post-processing for cleaner output.
        
        Args:
            submission_text (str): The student's submission text.
            expected_content (str): The expected content for the criterion.
            expected_content (str): The expected content for the criterion (used for context).
            score (float): The calculated score for the criterion.
            max_score (float): The maximum possible score for the criterion.
        
        Returns:
            str: Generated feedback.
        """
        # More nuanced prompt for feedback generation
        if score / max_score > 0.8:
            feedback_prompt = f"The submission demonstrates strong understanding of the topic. Specifically, regarding '{expected_content[:50]}...', the submission was well-articulated. Provide further suggestions for excellence. Feedback:"
        elif score / max_score > 0.5:
            feedback_prompt = f"The submission shows a good grasp of the topic, but there are areas for improvement. For '{expected_content[:50]}...', consider elaborating on the following. Feedback:"
        else:
            feedback_prompt = f"The submission needs significant improvement in understanding '{expected_content[:50]}...'. Please review the core concepts related to this criterion. Feedback:"

        # Generate feedback using the pre-trained model
        generated_feedback = self.feedback_generator_pipeline(
            feedback_prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            truncation=True
        )[0]['generated_text']
        log.debug(f"{Colors.BLUE}Generating feedback for score {score:.2f}/{max_score}...{Colors.RESET}")
        
        if max_score <= 0:
            return "Feedback generation skipped: Maximum score for this criterion is zero or invalid."

        score_ratio = score / max_score
        
        # Crafting more granular and empathetic prompt tones and suggestions
        if score_ratio >= 0.95:
            prompt_tone = "exceptional, demonstrating a deep understanding"
            suggestion_phrase = "To truly innovate and make a breakthrough, consider exploring:"
        elif score_ratio >= 0.85:
            prompt_tone = "very strong and well-developed"
            suggestion_phrase = "To elevate this further, you might refine or expand upon:"
        elif score_ratio >= 0.70:
            prompt_tone = "solid and comprehensive"
            suggestion_phrase = "To strengthen this area, focus on:"
        elif score_ratio >= 0.50:
            prompt_tone = "decent, covering the basics, but with significant room for improvement"
            suggestion_phrase = "To significantly enhance this section, prioritize:"
        elif score_ratio >= 0.25:
            prompt_tone = "emerging, indicating a foundational grasp but requiring substantial development"
            suggestion_phrase = "This area needs focused attention. Key aspects to review are:"
        else:
            prompt_tone = "minimal, suggesting core concepts may need revisiting"
            suggestion_phrase = "It is highly recommended to revisit the foundational principles and examples related to:"

        # Combine submission and expected content into a single context for the prompt
        # Limiting length to prevent exceeding model's context window
        context_len = 350 # Increased context length for richer snippets
        submission_snippet = submission_text[:context_len] + ("..." if len(submission_text) > context_len else "")
        expected_snippet = expected_content[:context_len] + ("..." if len(expected_content) > context_len else "")

        feedback_prompt = (
            f"As an empathetic, expert academic reviewer, provide constructive feedback for a student's submission. "
            f"The student's writing pertinent to this criterion is: \"{submission_snippet}\". "
            f"The ideal content expected for this criterion is: \"{expected_snippet}\". "
            f"The student achieved a {prompt_tone} score of {max_score - score:.2f} out of {max_score}. " # Changed to remaining points for improvement focus
            f"{suggestion_phrase} "
            f"Offer specific, actionable suggestions for improvement, or detailed praise for excellent areas. "
            f"Ensure the feedback is concise, encouraging, and directly addresses the content. Feedback:"
        )

        try:
            generated_feedback = self.feedback_generator_pipeline(
                feedback_prompt,
                max_new_tokens=180, # Increased max_new_tokens for more comprehensive feedback
                num_return_sequences=1,
                do_sample=True, # Enable sampling for more creative and varied feedback
                top_k=60,         # Top-k sampling for quality
                top_p=0.95,       # Nucleus sampling for diversity
                temperature=0.7,  # Controls randomness; 0.7 balances creativity and coherence
                pad_token_id=self.tokenizer.eos_token_id, # Prevents errors for padded inputs
                truncation=True, # Ensure prompt is truncated if too long for the model
            )[0]['generated_text']

            # Post-process: remove the prompt itself from the generated text
            if generated_feedback.startswith(feedback_prompt):
                generated_feedback = generated_feedback[len(feedback_prompt):].strip()
            
            # Advanced post-processing: remove incomplete sentences or model artifacts
            # Attempt to cut off at the last complete sentence if it's likely an incomplete generation
            sentences = re.split(r'(?<=[.!?])\s+', generated_feedback)
            if sentences and not generated_feedback.endswith(('.', '!', '?')):
                # If the last sentence is incomplete, try to use the second to last if it exists and is complete
                if len(sentences) > 1 and sentences[-2].endswith(('.', '!', '?')):
                    generated_feedback = ' '.join(sentences[:-1]).strip() + "."
                elif sentences[0].endswith(('.', '!', '?')): # If only one sentence, and it's complete
                    generated_feedback = sentences[0].strip()
                else: # If still incomplete, and no appropriate punctuation nearby, cut at last punctuation mark if present
                    last_punc_idx = generated_feedback.rfind('.')
                    if last_punc_idx == -1: # No full stop found
                        last_punc_idx = generated_feedback.rfind('!')
                    if last_punc_idx == -1: # No exclamation mark found
                        last_punc_idx = generated_feedback.rfind('?')
                    
                    if last_punc_idx != -1 and last_punc_idx > len(generated_feedback) * 0.7: # Only cut if near the end
                        generated_feedback = generated_feedback[:last_punc_idx + 1]
                    else: # If still incomplete, and no appropriate punctuation nearby, truncate at reasonable length
                        generated_feedback = generated_feedback[:min(len(generated_feedback), 150)].strip() + "..."
            
            # Remove any trailing newlines or extra spaces
            generated_feedback = generated_feedback.strip()
            
            log.debug(f"{Colors.GREEN}Feedback generated successfully.{Colors.RESET}")
            return generated_feedback

        except Exception as e:
            log.error(f"{Colors.RED}Failed to generate feedback for score {score:.2f}/{max_score}: {e}{Colors.RESET}")
            return f"Automated feedback generation failed due to an internal error. Please contact support. (Error: {e})"


if __name__ == "__main__":
    print("Run `pytest` to execute the tests.")
