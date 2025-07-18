import torch
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification  # Import Auto components
import logging
from datetime import datetime
import time
from typing import List

from utils.colors import Colors

# --- Logger Setup ---
class PremiumFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: f"{Colors.DIM}%(asctime)s | %(levelname)8s | %(name)s | %(message)s{Colors.RESET}",
        logging.INFO: f"{Colors.CYAN}%(asctime)s | %(levelname)8s |{Colors.RESET}{Colors.WHITE} %(name)s |{Colors.RESET} %(message)s{Colors.RESET}",
        logging.WARNING: f"{Colors.YELLOW}%(asctime)s | %(levelname)8s | %(name)s | %(message)s{Colors.RESET}",
        logging.ERROR: f"{Colors.RED}{Colors.BOLD}%(asctime)s | %(levelname)8s | %(name)s | %(message)s{Colors.RESET}",
        logging.CRITICAL: f"{Colors.BG_RED}{Colors.BRIGHT_WHITE}{Colors.BOLD}%(asctime)s | %(levelname)8s | %(name)s | %(message)s{Colors.RESET}"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

log = logging.getLogger("SourceEvaluator")
log.setLevel(logging.INFO)
if log.hasHandlers():
    log.handlers.clear()
ch = logging.StreamHandler()
ch.setFormatter(PremiumFormatter())
log.addHandler(ch)


class SourceEvaluator:
    """
    A class to evaluate the credibility, bias, and relevance of academic sources.
    Integrates with open academic search APIs and uses NLP for evaluation.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the SourceEvaluator with a specified device.
        Args:
            device (str): The device to use for computations ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        log.info(f"{Colors.BLUE}{Colors.BOLD}🚀 SourceEvaluator initialized on device: {self.device}{Colors.RESET}")

        pipeline_device_id = 0 if self.device.type == 'cuda' else -1

        log.info(f"{Colors.YELLOW}⚙️  Loading NLP models. This might take a moment...{Colors.RESET}")
        
        # --- Sentiment Analyzer (DistilBERT SST-2) ---
        log.info(f"{Colors.YELLOW}⚙️  Loading sentiment analyzer (distilbert-base-uncased-finetuned-sst-2-english)...{Colors.RESET}")
        try:
            # Explicitly load tokenizer and model using Auto components
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
            self.sentiment_model.to(self.device) # Move model to device
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
                tokenizer=self.sentiment_tokenizer,
                device=pipeline_device_id
            )
            log.info(f"{Colors.GREEN}✅ Sentiment analyzer loaded.{Colors.RESET}")
        except Exception as e:
            log.critical(f"{Colors.BG_RED}❌ CRITICAL: Failed to load sentiment analyzer: {e}{Colors.RESET}")
            self.sentiment_analyzer = None
            self.sentiment_tokenizer = None
            self.sentiment_model = None


        # --- Summarizer (DistilBART CNN) ---
        log.info(f"{Colors.YELLOW}⚙️  Loading summarizer (sshleifer/distilbart-cnn-12-6)...{Colors.RESET}")
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            log.info(f"{Colors.GREEN}✅ Summarizer loaded.{Colors.RESET}")
        except Exception as e:
            log.error(f"{Colors.RED}❌ Failed to load summarizer: {e}{Colors.RESET}")
            self.summarizer = None

        # --- Zero-Shot Classifier (BART Large MNLI) ---
        log.info(f"{Colors.YELLOW}⚙️  Loading zero-shot classifier (facebook/bart-large-mnli)...{Colors.RESET}")
        try:
            # Explicitly load tokenizer and model using Auto components for zero-shot
            self.zero_shot_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            self.zero_shot_model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
            self.zero_shot_model.to(self.device) # Move model to device
            self.zero_shot_classifier = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
                tokenizer=self.sentiment_tokenizer,
                device=pipeline_device_id
            )
            log.info(f"{Colors.GREEN}✅ Zero-shot classifier loaded.{Colors.RESET}")
        except Exception as e:
            log.error(f"{Colors.RED}❌ Failed to load zero-shot classifier: {e}{Colors.RESET}")
            self.zero_shot_classifier = None
            self.zero_shot_tokenizer = None
            self.zero_shot_model = None

        log.info(f"{Colors.BLUE}✨ SourceEvaluator initialization complete.{Colors.RESET}")

    def _fetch_content(self, identifier: str) -> str:
        log.info(f"{Colors.BRIGHT_BLUE}🌐 Attempting to fetch content for: {identifier}{Colors.RESET}")
        if identifier.startswith("http"):
            try:
                response = requests.get(identifier, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
                text_content = ' '.join([p.get_text() for p in paragraphs])
                log.info(f"{Colors.GREEN}✅ Content fetched successfully from URL.{Colors.RESET}")
                return text_content
            except requests.exceptions.RequestException as e:
                log.error(f"{Colors.RED}❌ Error fetching URL {identifier}: {e}{Colors.RESET}")
                return ""
        else:
            simulated_content = f"Simulated content for {identifier}: This is a research paper discussing advanced topics in AI and machine learning. It presents novel algorithms and experimental results. The authors are from a reputable institution. This paper was published in a peer-reviewed journal and has been cited by many other researchers. The research focuses on the intersection of neuroscience and artificial intelligence, exploring how biological neural networks can inspire more efficient and robust AI models. Experimental data suggests a significant improvement in learning rates and generalization capabilities when incorporating bio-inspired principles. This work contributes to the growing field of neuromorphic computing."
            log.info(f"{Colors.MAGENTA}💡 Simulated content generated for non-URL identifier.{Colors.RESET}")
            return simulated_content

    def evaluate_source(self, identifier: str, metadata: dict = None) -> dict:
        log.info(f"\n{Colors.BLUE}{Colors.BOLD}--- Initiating Source Evaluation for: {identifier} ---{Colors.RESET}")
        
        print(f"{Colors.DIM}  Processing source...{Colors.RESET}", end='\r')
        time.sleep(0.5)

        content = self._fetch_content(identifier)
        if not content:
            log.warning(f"{Colors.YELLOW}⚠️  No content retrieved for evaluation.{Colors.RESET}")
            return {"credibility": "N/A", "bias": "N/A", "relevance": "N/A", "summary": "Could not retrieve content."}

        # Adjusted for direct model use if pipeline failed to load
        if self.sentiment_model and self.sentiment_tokenizer:
            # When using AutoModel directly, you need to manage tokenization and model inference.
            inputs = self.sentiment_tokenizer(content, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Example interpretation for distilbert-base-uncased-finetuned-sst-2-english
            # Assumes LABEL_0 is negative, LABEL_1 is positive
            labels = ['NEGATIVE', 'POSITIVE']
            sentiment_result = [{'label': labels[predictions.argmax().item()], 'score': predictions.max().item()}]
            log.debug(f"{Colors.DIM}  Direct sentiment model computation performed. Output: {sentiment_result}{Colors.RESET}")
        elif self.sentiment_analyzer: # Fallback to pipeline if it somehow loaded differently
             sentiment_result = self.sentiment_analyzer(content[:512])
             log.debug(f"{Colors.DIM}  Pipeline sentiment analyzer computation performed. Output: {sentiment_result}{Colors.RESET}")
        else:
            sentiment_result = None
            log.warning(f"{Colors.YELLOW}⚠️  Sentiment analyzer not available, skipping sentiment computation.{Colors.RESET}")


        evaluation_results = {}

        # Credibility
        if self.zero_shot_classifier:
            log.info(f"{Colors.BRIGHT_CYAN}  Assessing credibility...{Colors.RESET}")
            credibility_labels = ["highly credible", "moderately credible", "low credibility"]
            try:
                credibility_results = self.zero_shot_classifier(content, credibility_labels, multi_label=False)
                credibility = credibility_results["labels"][0] if credibility_results else "N/A"
                evaluation_results["credibility"] = credibility
                log.info(f"{Colors.GREEN}  Credibility: {Colors.BOLD}{credibility}{Colors.RESET}")
            except Exception as e:
                log.error(f"{Colors.RED}❌ Error assessing credibility: {e}{Colors.RESET}")
                evaluation_results["credibility"] = "Error"
        else:
            log.warning(f"{Colors.YELLOW}⚠️  Zero-shot classifier not available, skipping credibility assessment.{Colors.RESET}")
            evaluation_results["credibility"] = "N/A (Classifier Unavailable)"

        # Bias
        bias_info = {}
        if self.sentiment_analyzer: # Now checks if sentiment_analyzer (pipeline) is available
            log.info(f"{Colors.BRIGHT_CYAN}  Analyzing sentiment for bias...{Colors.RESET}")
            try:
                bias_sentiment = "Neutral"
                if sentiment_result: # Use the result obtained above
                    label = sentiment_result[0]["label"]
                    score = sentiment_result[0]["score"]
                    if label == 'NEGATIVE' and score > 0.9: # This applies to 'distilbert-base-uncased-finetuned-sst-2-english'
                        bias_sentiment = f"{Colors.RED}Potentially Biased (Negative Tone){Colors.RESET}"
                    elif label == 'POSITIVE' and score > 0.9: # This applies to 'distilbert-base-uncased-finetuned-sst-2-english'
                        bias_sentiment = f"{Colors.GREEN}Potentially Biased (Positive Tone){Colors.RESET}"
                    else:
                        bias_sentiment = "Neutral"
                bias_info["Sentiment"] = bias_sentiment
                log.info(f"  Sentiment Bias: {bias_sentiment}{Colors.RESET}")
            except Exception as e:
                log.error(f"{Colors.RED}❌ Error analyzing sentiment: {e}{Colors.RESET}")
                bias_info["Sentiment"] = "Error"
        else:
            log.warning(f"{Colors.YELLOW}⚠️  Sentiment analyzer not available, skipping sentiment bias.{Colors.RESET}")
            bias_info["Sentiment"] = "N/A (Analyzer Unavailable)"


        if self.zero_shot_classifier:
            log.info(f"{Colors.BRIGHT_CYAN}  Analyzing political bias...{Colors.RESET}")
            political_bias_labels = ["left-leaning", "right-leaning", "neutral political bias"]
            try:
                # The zero_shot_classifier pipeline handles tokenization and inference internally
                political_bias_results = self.zero_shot_classifier(content, political_bias_labels, multi_label=False)
                political_bias = political_bias_results["labels"][0] if political_bias_results else "N/A"
                bias_info["Political"] = political_bias
                log.info(f"  Political Bias: {Colors.BOLD}{political_bias}{Colors.RESET}")
            except Exception as e:
                log.error(f"{Colors.RED}❌ Error analyzing political bias: {e}{Colors.RESET}")
                bias_info["Political"] = "Error"
        else:
            log.warning(f"{Colors.YELLOW}⚠️  Zero-shot classifier not available, skipping political bias.{Colors.RESET}")
            bias_info["Political"] = "N/A (Classifier Unavailable)"
        
        evaluation_results["bias"] = f"Sentiment: {bias_info.get('Sentiment', 'N/A')}, Political: {bias_info.get('Political', 'N/A')}"

        # Relevance
        log.info(f"{Colors.BRIGHT_CYAN}  Assessing relevance...{Colors.RESET}")
        relevance_score = len(content) / 1000.0
        if metadata and "keywords" in metadata:
            relevance_score += 0.5
            log.debug(f"{Colors.DIM}  Relevance boosted by keywords in metadata.{Colors.RESET}")
        
        summary = "No summary available."
        if self.summarizer:
            log.info(f"{Colors.BRIGHT_CYAN}  Generating summary for relevance...{Colors.RESET}")
            try:
                summary_result = self.summarizer(content, max_length=150, min_length=50, do_sample=False)
                summary = summary_result[0]["summary_text"] if summary_result else "No summary available."
                if len(summary.split()) > 20:
                    relevance_score += 0.3
                    log.debug(f"{Colors.DIM}  Relevance boosted by substantial summary.{Colors.RESET}")
                log.info(f"{Colors.GREEN}  Summary generated.{Colors.RESET}")
            except Exception as e:
                log.error(f"{Colors.RED}❌ Error generating summary: {e}{Colors.RESET}")
                summary = "Error generating summary."
        else:
            log.warning(f"{Colors.YELLOW}⚠️  Summarizer not available, skipping summary generation for relevance.{Colors.RESET}")

        relevance_text = "High" if relevance_score >= 1.5 else ("Medium" if relevance_score >= 0.5 else "Low")
        evaluation_results["relevance"] = relevance_text
        evaluation_results["summary"] = summary
        log.info(f"{Colors.GREEN}  Relevance: {Colors.BOLD}{relevance_text}{Colors.RESET}")

        print(f"\n{Colors.GREEN}{Colors.BOLD}--- Source Evaluation Complete ---{Colors.RESET}")
        self._present_evaluation_results(identifier, evaluation_results)
        return evaluation_results

    def _present_evaluation_results(self, identifier: str, results: dict):
        print(f"\n{Colors.BRIGHT_MAGENTA}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}║ {Colors.BOLD}EVALUATION REPORT FOR:{Colors.RESET} {Colors.WHITE}{identifier:<50.50}{Colors.RESET}{Colors.BRIGHT_MAGENTA} ║{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}╠═════════════════════════════════════════════════════════════════════════╣{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}║ {Colors.BOLD}Timestamp:{Colors.RESET} {Colors.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<60}{Colors.RESET}{Colors.BRIGHT_MAGENTA} ║{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}╟─────────────────────────────────────────────────────────────────────────╢{Colors.RESET}")
        
        credibility_color = Colors.GREEN if results.get("credibility") == "highly credible" else \
                            (Colors.YELLOW if results.get("credibility") == "moderately credible" else Colors.RED)
        print(f"{Colors.BRIGHT_MAGENTA}║ {Colors.BOLD}Credibility:{Colors.RESET} {credibility_color}{results.get('credibility', 'N/A'):<58}{Colors.RESET}{Colors.BRIGHT_MAGENTA} ║{Colors.RESET}")

        bias_color = Colors.CYAN if "Neutral" in results.get("bias", "") else \
                     (Colors.YELLOW if "Potentially Biased" in results.get("bias", "") else Colors.RED)
        print(f"{Colors.BRIGHT_MAGENTA}║ {Colors.BOLD}Bias Analysis:{Colors.RESET} {bias_color}{results.get('bias', 'N/A'):<58}{Colors.RESET}{Colors.BRIGHT_MAGENTA} ║{Colors.RESET}")

        relevance_color = Colors.GREEN if results.get("relevance") == "High" else \
                          (Colors.YELLOW if results.get("relevance") == "Medium" else Colors.RED)
        print(f"{Colors.BRIGHT_MAGENTA}║ {Colors.BOLD}Relevance:   {Colors.RESET} {relevance_color}{results.get('relevance', 'N/A'):<58}{Colors.RESET}{Colors.BRIGHT_MAGENTA} ║{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_MAGENTA}╟─────────────────────────────────────────────────────────────────────────╢{Colors.RESET}")
        summary_lines = [results.get('summary', 'N/A')[i:i+70] for i in range(0, len(results.get('summary', 'N/A')), 70)]
        print(f"{Colors.BRIGHT_MAGENTA}║ {Colors.BOLD}Summary:{Colors.RESET} {Colors.DIM}{summary_lines[0]:<62}{Colors.RESET}{Colors.BRIGHT_MAGENTA} ║{Colors.RESET}")
        for line in summary_lines[1:]:
            print(f"{Colors.BRIGHT_MAGENTA}║ {'':<9}{Colors.DIM}{line:<62}{Colors.RESET}{Colors.BRIGHT_MAGENTA} ║{Colors.RESET}")

        print(f"{Colors.BRIGHT_MAGENTA}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")

    def search_academic_api(self, query: str, api_type: str = "arxiv") -> list:
        log.info(f"\n{Colors.BLUE}{Colors.BOLD}--- Initiating Academic Search for '{query}' on {api_type.upper()} ---{Colors.RESET}")
        results = []
        if api_type == "arxiv":
            arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
            log.info(f"{Colors.DIM}  Querying arXiv: {arxiv_url}{Colors.RESET}")
            try:
                response = requests.get(arxiv_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'xml')
                for entry in soup.find_all('entry'):
                    title = entry.title.text.strip()
                    link = entry.link['href']
                    authors = [author.find('name').text for author in entry.find_all('author')]
                    results.append({"title": title, "url": link, "authors": authors})
                log.info(f"{Colors.GREEN}✅ Retrieved {len(results)} results from arXiv.{Colors.RESET}")
            except requests.exceptions.RequestException as e:
                log.error(f"{Colors.RED}❌ Error searching arXiv: {e}{Colors.RESET}")
            dummy_tensor = torch.randn(5, 5, device=self.device)
            _ = torch.linalg.det(dummy_tensor)
            log.debug(f"{Colors.DIM}  Dummy PyTorch computation performed (arXiv search).{Colors.RESET}")

        elif api_type == "semantic_scholar":
            semantic_scholar_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,url,authors"
            log.info(f"{Colors.DIM}  Querying Semantic Scholar: {semantic_scholar_url}{Colors.RESET}")
            try:
                response = requests.get(semantic_scholar_url, timeout=15)
                response.raise_for_status()
                data = response.json()
                for paper in data.get('data', []):
                    title = paper.get('title')
                    url = paper.get('url')
                    authors = [author.get('name') for author in paper.get('authors', [])]
                    results.append({"title": title, "url": url, "authors": authors})
                log.info(f"{Colors.GREEN}✅ Retrieved {len(results)} results from Semantic Scholar.{Colors.RESET}")
            except requests.exceptions.RequestException as e:
                log.error(f"{Colors.RED}❌ Error searching Semantic Scholar: {e}{Colors.RESET}")
            dummy_tensor = torch.randn(7, 7, device=self.device)
            _ = torch.mean(dummy_tensor)
            log.debug(f"{Colors.DIM}  Dummy PyTorch computation performed (Semantic Scholar search).{Colors.RESET}")
        
        print(f"\n{Colors.BLUE}{Colors.BOLD}--- Academic Search Complete ---{Colors.RESET}")
        self._present_search_results(query, api_type, results)
        return results

    def _present_search_results(self, query: str, api_type: str, results: list):
        print(f"\n{Colors.BRIGHT_BLUE}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}║ {Colors.BOLD}SEARCH RESULTS FOR:{Colors.RESET} {Colors.WHITE}'{query}' on {api_type.upper()}{Colors.RESET}{Colors.BRIGHT_BLUE:<45.45} ║{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}╠═════════════════════════════════════════════════════════════════════════╣{Colors.RESET}")
        if not results:
            print(f"{Colors.BRIGHT_BLUE}║ {Colors.YELLOW}No results found for this query.{Colors.RESET}{'':<62}{Colors.BRIGHT_BLUE} ║{Colors.RESET}")
        else:
            for i, result in enumerate(results):
                title = result.get('title', 'N/A')
                url = result.get('url', 'N/A')
                authors = ", ".join(result.get('authors', ['N/A']))
                
                print(f"{Colors.BRIGHT_BLUE}║ {Colors.BOLD}Result {i+1}:{'':<67}{Colors.BRIGHT_BLUE} ║{Colors.RESET}")
                print(f"{Colors.BRIGHT_BLUE}║   {Colors.CYAN}Title:{Colors.RESET} {Colors.WHITE}{title:<62.62}{Colors.RESET}{Colors.BRIGHT_BLUE} ║{Colors.RESET}")
                print(f"{Colors.BRIGHT_BLUE}║   {Colors.CYAN}URL:  {Colors.RESET} {Colors.UNDERLINE}{Colors.DIM}{url:<62.62}{Colors.RESET}{Colors.BRIGHT_BLUE} ║{Colors.RESET}")
                print(f"{Colors.BRIGHT_BLUE}║   {Colors.CYAN}Authors:{Colors.RESET} {Colors.DIM}{authors:<62.62}{Colors.RESET}{Colors.BRIGHT_BLUE} ║{Colors.RESET}")
                if i < len(results) - 1:
                    print(f"{Colors.BRIGHT_BLUE}╟─────────────────────────────────────────────────────────────────────────╢{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")

    def fetch_top_n(self, query: str, n: int = 5) -> List[str]:
        log.info(f"{Colors.BRIGHT_BLUE}📦 Fetching top {n} sources for '{query}'...{Colors.RESET}")
        
        raw_results = self.search_academic_api(query, api_type="arxiv")
        
        top_n_prompts = []
        for i, result in enumerate(raw_results[:n]):
            title = result.get("title", "Unknown Title")
            url = result.get("url", "No URL")
            authors = ", ".join(result.get("authors", []))
            
            prompt_string = (
                f"Source {i+1}: \"{title}\" "
                f"(Authors: {authors if authors else 'N/A'}) "
                f"URL: {url}"
            )
            top_n_prompts.append(prompt_string)
            log.debug(f"{Colors.DIM}  Generated prompt for source {i+1}.{Colors.RESET}")
            
        if not top_n_prompts:
            log.warning(f"{Colors.YELLOW}⚠️  No top sources found for '{query}'. Returning empty list.{Colors.RESET}")
        else:
            log.info(f"{Colors.GREEN}✅ Successfully fetched and formatted {len(top_n_prompts)} top sources.{Colors.RESET}")
            
        return top_n_prompts


# --- Main Execution Block ---
if __name__ == "__main__":
    log.info(f"{Colors.BLUE}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BLUE}{Colors.BOLD}║ {Colors.BRIGHT_WHITE}INITIATING NEURORESEARCHER SOURCE EVALUATION MODULE{Colors.RESET}{Colors.BLUE}{Colors.BOLD:<10} ║{Colors.RESET}")
    log.info(f"{Colors.BLUE}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    if torch.cuda.is_available():
        device = 'cuda'
        log.info(f"{Colors.GREEN}{Colors.BOLD}✨ CUDA is available! Utilizing GPU acceleration. ✨{Colors.RESET}")
    else:
        device = 'cpu'
        log.warning(f"{Colors.YELLOW}{Colors.BOLD}⚠️ CUDA not available. Running on CPU. Performance may be impacted. ⚠️{Colors.RESET}")

    evaluator = SourceEvaluator(device=device)

    url_to_evaluate = "https://www.nature.com/articles/d41586-024-01041-3"
    evaluation_results = evaluator.evaluate_source(url_to_evaluate)

    pdf_identifier = "synthetic_neuro_paper.pdf"
    pdf_metadata = {"author": "Dr. A. Synapse", "journal": "Journal of Neuromorphic AI", "keywords": ["neuromorphic", "AI ethics", "brain-computer interfaces"]}
    pdf_evaluation_results = evaluator.evaluate_source(pdf_identifier, metadata=pdf_metadata)

    search_query = "quantum computing in neuroscience"
    arxiv_results = evaluator.search_academic_api(search_query, api_type="arxiv")

    semantic_scholar_results = evaluator.search_academic_api(search_query, api_type="semantic_scholar")

    print(f"\n{Colors.BLUE}{Colors.BOLD}--- Testing fetch_top_n ---{Colors.RESET}")
    top_prompts = evaluator.fetch_top_n("impact of climate change on health", n=3)
    for i, prompt in enumerate(top_prompts):
        print(f"{Colors.CYAN}  [{i+1}] {prompt}{Colors.RESET}")
    if not top_prompts:
        print(f"{Colors.YELLOW}  No prompts were generated by fetch_top_n.{Colors.RESET}")


    log.info(f"\n{Colors.BLUE}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BLUE}{Colors.BOLD}║ {Colors.BRIGHT_WHITE}SOURCE EVALUATION MODULE TERMINATED.{Colors.RESET}{Colors.BLUE}{Colors.BOLD:<34} ║{Colors.RESET}")
    log.info(f"{Colors.BLUE}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")