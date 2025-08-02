import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from utils.tensor_ops import tensor_to_ndarray
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import faiss
import pickle
import threading
import time
from queue import Queue
import sqlite3
from contextlib import contextmanager

@dataclass
class RealtimeContext:
    """Container for real-time contextual information"""
    timestamp: datetime
    user_query: str
    retrieved_docs: List[str]
    topic_confidence: float
    relevance_score: float
    metadata: Dict[str, Any]

class RAGRetriever:
    """Retrieval-Augmented Generation component for context gathering"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.index = None
        self.documents = []
        self.embeddings_cache = {}
        
    def build_index(self, documents: List[str], save_path: Optional[str] = None):
        """Build FAISS index from documents"""
        embeddings = []
        for doc in documents:
            embedding = self._get_embedding(doc)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents = documents
        
        if save_path:
            faiss.write_index(self.index, f"{save_path}.index")
            with open(f"{save_path}_docs.pkl", "wb") as f:
                pickle.dump(documents, f)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embedding = tensor_to_ndarray(outputs.last_hidden_state.mean(dim=1).squeeze())
        
        self.embeddings_cache[text] = embedding
        return embedding
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for query"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self._get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results

class ChainOfThoughtReasoner:
    """Chain-of-thought reasoning component using a smaller model"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def reason(self, context: RealtimeContext, target_task: str) -> Dict[str, Any]:
        """Perform chain-of-thought reasoning for label generation"""
        
        # Construct reasoning prompt
        prompt = self._build_reasoning_prompt(context, target_task)
        
        # Generate reasoning steps
        reasoning_steps = self._generate_reasoning(prompt)
        
        # Extract final decision
        decision = self._extract_decision(reasoning_steps)
        
        return {
            "reasoning_steps": reasoning_steps,
            "decision": decision,
            "confidence": self._calculate_confidence(reasoning_steps, decision),
            "metadata": {
                "context_relevance": context.relevance_score,
                "reasoning_length": len(reasoning_steps),
                "timestamp": context.timestamp
            }
        }
    
    def _build_reasoning_prompt(self, context: RealtimeContext, task: str) -> str:
        """Build chain-of-thought prompt"""
        retrieved_context = "\n".join(context.retrieved_docs[:3])
        
        prompt = f"""
        Task: {task}
        User Query: {context.user_query}
        Retrieved Context: {retrieved_context}
        Topic Confidence: {context.topic_confidence:.2f}
        
        Let me think step by step:
        1. What is the main topic/theme?
        2. What does the context suggest?
        3. How confident can I be in this assessment?
        4. What would be the most appropriate label?
        
        Reasoning:
        """
        
        return prompt
    
    def _generate_reasoning(self, prompt: str) -> List[str]:
        """Generate reasoning steps"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + 200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning_text = generated_text[len(prompt):].strip()
        
        # Split into steps
        steps = [step.strip() for step in reasoning_text.split('\n') if step.strip()]
        return steps
    
    def _extract_decision(self, reasoning_steps: List[str]) -> str:
        """Extract final decision from reasoning steps"""
        # Look for decision keywords in the last few steps
        decision_keywords = ["label:", "decision:", "conclusion:", "answer:"]
        
        for step in reversed(reasoning_steps):
            for keyword in decision_keywords:
                if keyword in step.lower():
                    return step.split(keyword)[-1].strip()
        
        # Fallback: return the last step
        return reasoning_steps[-1] if reasoning_steps else "unknown"
    
    def _calculate_confidence(self, reasoning_steps: List[str], decision: str) -> float:
        """Calculate confidence based on reasoning quality"""
        # Simple heuristic: longer reasoning = higher confidence
        base_confidence = min(0.9, len(reasoning_steps) * 0.15)
        
        # Boost confidence if decision contains confident language
        confident_words = ["clearly", "definitely", "obviously", "certainly"]
        if any(word in decision.lower() for word in confident_words):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

class RealtimeDataStream:
    """Real-time data streaming component"""
    
    def __init__(self, update_interval: int = 60):
        self.update_interval = update_interval
        self.data_queue = Queue()
        self.is_running = False
        self.thread = None
        self.data_sources = []
        
    def add_data_source(self, source_func):
        """Add a data source function"""
        self.data_sources.append(source_func)
    
    def start_streaming(self):
        """Start the real-time data stream"""
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_loop)
        self.thread.start()
    
    def stop_streaming(self):
        """Stop the real-time data stream"""
        self.is_running = False
        if self.thread:
            self.thread.join()
    
    def _stream_loop(self):
        """Main streaming loop"""
        while self.is_running:
            try:
                # Collect data from all sources
                for source in self.data_sources:
                    try:
                        data = source()
                        if data:
                            self.data_queue.put(data)
                    except Exception as e:
                        logging.error(f"Error in data source: {e}")
                
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in streaming loop: {e}")
    
    def get_latest_data(self) -> List[Any]:
        """Get latest data from the stream"""
        data = []
        while not self.data_queue.empty():
            data.append(self.data_queue.get())
        return data

class LabelQualityAssessor:
    """Assess the quality of generated labels"""
    
    def __init__(self):
        self.historical_accuracy = {}
        self.confidence_threshold = 0.7
        
    def assess_label_quality(self, label: str, context: RealtimeContext, 
                           reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of a generated label"""
        
        quality_score = 0.0
        factors = {}
        
        # Factor 1: Context relevance
        context_score = context.relevance_score
        factors["context_relevance"] = context_score
        quality_score += context_score * 0.3
        
        # Factor 2: Reasoning confidence
        reasoning_confidence = reasoning["confidence"]
        factors["reasoning_confidence"] = reasoning_confidence
        quality_score += reasoning_confidence * 0.25
        
        # Factor 3: Topic confidence
        topic_confidence = context.topic_confidence
        factors["topic_confidence"] = topic_confidence
        quality_score += topic_confidence * 0.2
        
        # Factor 4: Reasoning depth
        reasoning_depth = len(reasoning["reasoning_steps"]) / 10.0  # Normalize
        factors["reasoning_depth"] = min(1.0, reasoning_depth)
        quality_score += factors["reasoning_depth"] * 0.15
        
        # Factor 5: Historical accuracy (if available)
        if label in self.historical_accuracy:
            hist_acc = self.historical_accuracy[label]
            factors["historical_accuracy"] = hist_acc
            quality_score += hist_acc * 0.1
        
        quality_score = min(1.0, quality_score)
        
        return {
            "quality_score": quality_score,
            "factors": factors,
            "is_high_quality": quality_score >= self.confidence_threshold,
            "recommendations": self._generate_recommendations(factors)
        }
    
    def _generate_recommendations(self, factors: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving label quality"""
        recommendations = []
        
        if factors["context_relevance"] < 0.5:
            recommendations.append("Improve context retrieval - low relevance")
        
        if factors["reasoning_confidence"] < 0.6:
            recommendations.append("Enhance reasoning model - low confidence")
        
        if factors["topic_confidence"] < 0.6:
            recommendations.append("Better topic detection needed")
        
        return recommendations

class CustomLabelsDataset(Dataset):
    """Main CustomLabels class that replaces traditional datasets"""
    
    def __init__(self, 
                 initial_data: List[str],
                 label_types: List[str],
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 reasoning_model: str = "microsoft/DialoGPT-small",
                 cache_size: int = 10000,
                 quality_threshold: float = 0.7):
        
        # Initialize components
        self.rag_retriever = RAGRetriever(embedding_model)
        self.reasoner = ChainOfThoughtReasoner(reasoning_model)
        self.data_stream = RealtimeDataStream()
        self.quality_assessor = LabelQualityAssessor()
        
        # Configuration
        self.label_types = label_types
        self.quality_threshold = quality_threshold
        self.cache_size = cache_size
        
        # Storage
        self.data_items = initial_data
        self.label_cache = {}
        self.quality_cache = {}
        
        # Database for persistence
        self.db_path = "custom_labels.db"
        self._init_database()
        
        # Build initial index
        self.rag_retriever.build_index(initial_data)
        
        # Start real-time streaming
        self._setup_data_sources()
        self.data_stream.start_streaming()
        
        logging.info(f"CustomLabelsDataset initialized with {len(initial_data)} items")
    
    def _init_database(self):
        """Initialize SQLite database for persistence"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY,
                    data_item TEXT,
                    label TEXT,
                    quality_score REAL,
                    timestamp DATETIME,
                    context_data TEXT
                )
            """)
            conn.commit()
    
    def _setup_data_sources(self):
        """Setup real-time data sources"""
        def sample_news_source():
            # Placeholder for real news API integration
            return f"Breaking: Recent development in AI research - {datetime.now()}"
        
        def sample_social_source():
            # Placeholder for social media API integration
            return f"Trending topic: Machine learning applications - {datetime.now()}"
        
        self.data_stream.add_data_source(sample_news_source)
        self.data_stream.add_data_source(sample_social_source)
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with real-time intelligent labeling"""
        if idx >= len(self.data_items):
            raise IndexError("Index out of range")
        
        data_item = self.data_items[idx]
        
        # Check cache first
        if data_item in self.label_cache:
            cached_result = self.label_cache[data_item]
            if self._is_cache_valid(cached_result):
                return cached_result
        
        # Generate real-time label
        label_result = self._generate_realtime_label(data_item, idx)
        
        # Cache if high quality
        if label_result["quality"]["is_high_quality"]:
            self.label_cache[data_item] = label_result
            self._persist_to_database(data_item, label_result)
        
        return label_result
    
    def _generate_realtime_label(self, data_item: str, idx: int) -> Dict[str, Any]:
        """Generate label using real-time context and reasoning"""
        
        # Get real-time context
        context = self._build_realtime_context(data_item)
        
        # Perform chain-of-thought reasoning
        reasoning_result = self.reasoner.reason(context, f"classify into: {self.label_types}")
        
        # Extract label from reasoning
        predicted_label = self._extract_label_from_reasoning(reasoning_result)
        
        # Assess quality
        quality_assessment = self.quality_assessor.assess_label_quality(
            predicted_label, context, reasoning_result
        )
        
        return {
            "data": data_item,
            "label": predicted_label,
            "context": context,
            "reasoning": reasoning_result,
            "quality": quality_assessment,
            "timestamp": datetime.now(),
            "index": idx
        }
    
    def _build_realtime_context(self, data_item: str) -> RealtimeContext:
        """Build real-time context for data item"""
        
        # Get latest real-time data
        latest_data = self.data_stream.get_latest_data()
        
        # Combine with data item for retrieval
        query = f"{data_item} {' '.join(latest_data[-3:])}"  # Use last 3 updates
        
        # Retrieve relevant context
        retrieved_docs = self.rag_retriever.retrieve(query, top_k=5)
        
        # Calculate relevance and topic confidence
        relevance_score = np.mean([score for _, score in retrieved_docs]) if retrieved_docs else 0.0
        topic_confidence = self._calculate_topic_confidence(data_item, retrieved_docs)
        
        return RealtimeContext(
            timestamp=datetime.now(),
            user_query=query,
            retrieved_docs=[doc for doc, _ in retrieved_docs],
            topic_confidence=topic_confidence,
            relevance_score=relevance_score,
            metadata={"latest_updates": latest_data[-5:]}
        )
    
    def _calculate_topic_confidence(self, data_item: str, retrieved_docs: List[Tuple[str, float]]) -> float:
        """Calculate confidence in topic detection"""
        if not retrieved_docs:
            return 0.0
        
        # Simple heuristic: average retrieval score
        scores = [score for _, score in retrieved_docs]
        return float(np.mean(scores))
    
    def _extract_label_from_reasoning(self, reasoning_result: Dict[str, Any]) -> str:
        """Extract the most appropriate label from reasoning"""
        decision = reasoning_result["decision"].lower()
        
        # Look for label types in the decision
        for label_type in self.label_types:
            if label_type.lower() in decision:
                return label_type
        
        # Fallback: return the label type with highest confidence
        return self.label_types[0]  # Default to first label type
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        cache_time = cached_result["timestamp"]
        return (datetime.now() - cache_time).seconds < 300  # 5 minutes
    
    def _persist_to_database(self, data_item: str, label_result: Dict[str, Any]):
        """Persist high-quality labels to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO labels (data_item, label, quality_score, timestamp, context_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                data_item,
                label_result["label"],
                label_result["quality"]["quality_score"],
                label_result["timestamp"],
                json.dumps(label_result["context"].__dict__, default=str)
            ))
            conn.commit()
    
    def add_data(self, new_data: List[str]):
        """Add new data to the dataset"""
        self.data_items.extend(new_data)
        # Rebuild index with new data
        self.rag_retriever.build_index(self.data_items)
        logging.info(f"Added {len(new_data)} new items. Dataset size: {len(self.data_items)}")
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality statistics for generated labels"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT AVG(quality_score), COUNT(*) FROM labels")
            avg_quality, count = cursor.fetchone()
        
        return {
            "average_quality": avg_quality or 0.0,
            "total_labels": count or 0,
            "cache_size": len(self.label_cache),
            "high_quality_ratio": self._calculate_high_quality_ratio()
        }
    
    def _calculate_high_quality_ratio(self) -> float:
        """Calculate ratio of high-quality labels"""
        if not self.label_cache:
            return 0.0
        
        high_quality_count = sum(1 for result in self.label_cache.values() 
                                if result["quality"]["is_high_quality"])
        return high_quality_count / len(self.label_cache)
    
    def export_labels(self, filepath: str):
        """Export generated labels to file"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM labels ORDER BY timestamp DESC")
            labels = cursor.fetchall()
        
        export_data = []
        for label_row in labels:
            export_data.append({
                "data_item": label_row[1],
                "label": label_row[2],
                "quality_score": label_row[3],
                "timestamp": label_row[4]
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logging.info(f"Exported {len(export_data)} labels to {filepath}")
    
    def cleanup(self):
        """Clean up resources"""
        self.data_stream.stop_streaming()
        logging.info("CustomLabelsDataset cleaned up")

# Usage example and testing
def create_sample_dataset():
    """Create a sample dataset for testing"""
    
    # Sample data
    sample_data = [
        "AI breakthrough in natural language processing",
        "Stock market volatility concerns investors",
        "Climate change impacts on agriculture",
        "New smartphone features announced",
        "Healthcare AI improves diagnosis accuracy",
        "Cryptocurrency market shows mixed signals",
        "Space exploration mission launches successfully",
        "Renewable energy adoption increases globally"
    ]
    
    # Label types
    label_types = ["technology", "finance", "environment", "health", "space", "energy"]
    
    # Create dataset
    dataset = CustomLabelsDataset(
        initial_data=sample_data,
        label_types=label_types,
        quality_threshold=0.7
    )
    
    return dataset

# Advanced usage with PyTorch DataLoader
def create_dataloader(dataset: CustomLabelsDataset, batch_size: int = 4):
    """Create DataLoader that works with CustomLabelsDataset"""
    
    def collate_fn(batch):
        """Custom collate function for real-time labeled data"""
        data_items = []
        labels = []
        contexts = []
        quality_scores = []
        
        for item in batch:
            data_items.append(item["data"])
            labels.append(item["label"])
            contexts.append(item["context"])
            quality_scores.append(item["quality"]["quality_score"])
        
        return {
            "data": data_items,
            "labels": labels,
            "contexts": contexts,
            "quality_scores": quality_scores
        }
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = create_sample_dataset()
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=2)
    
    # Test the dataset
    print("Testing CustomLabelsDataset...")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        for j, (data, label, quality) in enumerate(zip(batch["data"], batch["labels"], batch["quality_scores"])):
            print(f"  Item {j+1}: {data[:50]}...")
            print(f"  Label: {label}")
            print(f"  Quality: {quality:.3f}")
        
        if i >= 2:  # Only show first 3 batches
            break
    
    # Show quality statistics
    stats = dataset.get_quality_statistics()
    print(f"\nQuality Statistics:")
    print(f"Average Quality: {stats['average_quality']:.3f}")
    print(f"Total Labels: {stats['total_labels']}")
    print(f"High Quality Ratio: {stats['high_quality_ratio']:.3f}")
    
    # Export labels
    dataset.export_labels("generated_labels.json")
    
    # Cleanup
    dataset.cleanup()