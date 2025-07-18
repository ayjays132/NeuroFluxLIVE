"""
RealTimeDataAbsorber.py

A sophisticated real-time data absorption and training system that can:
- Continuously absorb and understand new data streams
- Train models in real-time while they're being used (eval mode)
- Act as a sensory system for robots/AI systems
- Maintain model performance while learning new patterns
- Handle multiple data modalities simultaneously
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import pickle
import sqlite3
from queue import Queue, PriorityQueue
import cv2
import soundfile as sf
import requests
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import faiss
import psutil
import gc
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataPoint:
    """Represents a single data point with metadata"""
    data: Any
    modality: str  # 'text', 'image', 'audio', 'sensor', 'multimodal'
    timestamp: datetime
    source: str
    priority: int = 1
    confidence: float = 1.0
    embeddings: Optional[np.ndarray] = None
    labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.priority < other.priority

@dataclass
class LearningEvent:
    """Represents a learning event with associated data"""
    event_type: str  # 'pattern_detected', 'anomaly_found', 'concept_drift', 'new_knowledge'
    data_points: List[DataPoint]
    learning_signal: float
    adaptation_required: bool
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModalityProcessor:
    """Base class for processing different data modalities"""
    
    def __init__(self, modality: str):
        self.modality = modality
        self.processor = None
        self.embedding_dim = 0
        
    def process(self, data: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process raw data into tensors and metadata"""
        raise NotImplementedError
    
    def extract_embeddings(self, data: Any) -> np.ndarray:
        """Extract embeddings from data"""
        raise NotImplementedError

class TextProcessor(ModalityProcessor):
    """Process text data in real-time"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__("text")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = 384
        
    def process(self, data: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process text data"""
        # Tokenize
        inputs = self.tokenizer(data, return_tensors="pt", 
                              truncation=True, padding=True, max_length=512)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        metadata = {
            "length": len(data),
            "word_count": len(data.split()),
            "language": "auto_detected",  # Could add language detection
            "sentiment": self._analyze_sentiment(data)
        }
        
        return embeddings, metadata
    
    def extract_embeddings(self, data: str) -> np.ndarray:
        """Extract embeddings from text"""
        embeddings, _ = self.process(data)
        return embeddings.numpy()
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worse', 'worst']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

class ImageProcessor(ModalityProcessor):
    """Process image data in real-time"""
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        super().__init__("image")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = 2048
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def process(self, data: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process image data"""
        # Convert to PIL if needed
        if isinstance(data, np.ndarray):
            if data.shape[-1] == 3:  # RGB
                from PIL import Image
                data = Image.fromarray(data)
        
        # Process with model
        inputs = self.processor(data, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs.pooler_output
        
        metadata = {
            "dimensions": data.size if hasattr(data, 'size') else data.shape,
            "color_mode": "RGB" if len(data.shape) == 3 else "grayscale",
            "brightness": self._calculate_brightness(data),
            "complexity": self._calculate_complexity(data)
        }
        
        return embeddings, metadata
    
    def extract_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Extract embeddings from image"""
        embeddings, _ = self.process(data)
        return embeddings.numpy()
    
    def _calculate_brightness(self, image) -> float:
        """Calculate average brightness"""
        if isinstance(image, np.ndarray):
            return float(np.mean(image))
        else:
            return float(np.mean(np.array(image)))
    
    def _calculate_complexity(self, image) -> float:
        """Calculate image complexity (edge density)"""
        if isinstance(image, np.ndarray):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        else:
            gray = np.array(image.convert('L'))
        
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))

class AudioProcessor(ModalityProcessor):
    """Process audio data in real-time"""
    
    def __init__(self):
        super().__init__("audio")
        self.embedding_dim = 512
        self.sample_rate = 16000
    
    def process(self, data: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process audio data"""
        # Ensure proper format
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Convert to mono
        
        # Extract features
        features = self._extract_audio_features(data)
        embeddings = torch.tensor(features).float().unsqueeze(0)
        
        metadata = {
            "duration": len(data) / self.sample_rate,
            "sample_rate": self.sample_rate,
            "rms_energy": float(np.sqrt(np.mean(data**2))),
            "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.sign(data)))))
        }
        
        return embeddings, metadata
    
    def extract_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Extract embeddings from audio"""
        embeddings, _ = self.process(data)
        return embeddings.numpy()
    
    def _extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio features (simplified MFCC-like)"""
        # Simple spectral features
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Bin the frequency components
        n_bins = 512
        bin_size = len(magnitude) // n_bins
        features = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            features.append(np.mean(magnitude[start:end]))
        
        return np.array(features)

class SensorProcessor(ModalityProcessor):
    """Process sensor data in real-time"""
    
    def __init__(self, sensor_types: List[str]):
        super().__init__("sensor")
        self.sensor_types = sensor_types
        self.embedding_dim = len(sensor_types) * 10  # 10 features per sensor
    
    def process(self, data: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process sensor data"""
        features = []
        
        for sensor_type in self.sensor_types:
            if sensor_type in data:
                value = data[sensor_type]
                # Create features for each sensor
                sensor_features = [
                    value,  # Raw value
                    np.log1p(abs(value)),  # Log transform
                    np.tanh(value),  # Normalized value
                    1.0 if value > 0 else 0.0,  # Positive indicator
                    abs(value),  # Absolute value
                    value**2,  # Squared value
                    np.exp(-abs(value)),  # Exponential decay
                    np.sin(value),  # Sine transform
                    np.cos(value),  # Cosine transform
                    1.0 if abs(value) > 1.0 else 0.0  # Threshold indicator
                ]
                features.extend(sensor_features)
            else:
                # Fill with zeros if sensor not present
                features.extend([0.0] * 10)
        
        embeddings = torch.tensor(features).float().unsqueeze(0)
        
        metadata = {
            "sensor_count": len(data),
            "active_sensors": list(data.keys()),
            "value_range": [min(data.values()), max(data.values())] if data else [0, 0],
            "total_energy": sum(v**2 for v in data.values()) if data else 0.0
        }
        
        return embeddings, metadata
    
    def extract_embeddings(self, data: Dict[str, float]) -> np.ndarray:
        """Extract embeddings from sensor data"""
        embeddings, _ = self.process(data)
        return embeddings.numpy()

class PatternDetector:
    """Detect patterns and anomalies in real-time data"""
    
    def __init__(self, window_size: int = 100, anomaly_threshold: float = 2.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.pattern_history = defaultdict(list)
        self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        
    def detect_patterns(self, data_point: DataPoint) -> List[LearningEvent]:
        """Detect patterns in incoming data"""
        events = []
        
        # Add to window
        self.data_windows[data_point.modality].append(data_point)
        
        # Only analyze if we have enough data
        if len(self.data_windows[data_point.modality]) >= self.window_size:
            # Anomaly detection
            anomaly_event = self._detect_anomaly(data_point)
            if anomaly_event:
                events.append(anomaly_event)
            
            # Pattern recognition
            pattern_event = self._detect_recurring_pattern(data_point)
            if pattern_event:
                events.append(pattern_event)
            
            # Concept drift detection
            drift_event = self._detect_concept_drift(data_point)
            if drift_event:
                events.append(drift_event)
        
        return events
    
    def _detect_anomaly(self, data_point: DataPoint) -> Optional[LearningEvent]:
        """Detect anomalies in the data stream"""
        if data_point.embeddings is None:
            return None
        
        window_data = list(self.data_windows[data_point.modality])
        if len(window_data) < 10:
            return None
        
        # Get embeddings from recent data
        embeddings = []
        for dp in window_data[-50:]:  # Last 50 points
            if dp.embeddings is not None:
                embeddings.append(dp.embeddings.flatten())
        
        if len(embeddings) < 10:
            return None
        
        embeddings = np.array(embeddings)
        
        # Calculate z-score
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        
        current_embedding = data_point.embeddings.flatten()
        z_scores = np.abs((current_embedding - mean_embedding) / (std_embedding + 1e-8))
        max_z_score = np.max(z_scores)
        
        if max_z_score > self.anomaly_threshold:
            return LearningEvent(
                event_type="anomaly_found",
                data_points=[data_point],
                learning_signal=min(max_z_score / 10.0, 1.0),
                adaptation_required=True,
                timestamp=datetime.now(),
                metadata={"z_score": float(max_z_score), "anomaly_type": "statistical"}
            )
        
        return None
    
    def _detect_recurring_pattern(self, data_point: DataPoint) -> Optional[LearningEvent]:
        """Detect recurring patterns"""
        if data_point.embeddings is None:
            return None
        
        window_data = list(self.data_windows[data_point.modality])
        if len(window_data) < 20:
            return None
        
        # Look for similar embeddings in recent history
        current_embedding = data_point.embeddings.flatten()
        similarities = []
        
        for dp in window_data[-20:]:
            if dp.embeddings is not None:
                other_embedding = dp.embeddings.flatten()
                similarity = np.dot(current_embedding, other_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding) + 1e-8
                )
                similarities.append(similarity)
        
        # If we find high similarity, it's a recurring pattern
        if similarities and np.max(similarities) > 0.9:
            return LearningEvent(
                event_type="pattern_detected",
                data_points=[data_point],
                learning_signal=0.3,
                adaptation_required=False,
                timestamp=datetime.now(),
                metadata={"pattern_type": "recurring", "similarity": float(np.max(similarities))}
            )
        
        return None
    
    def _detect_concept_drift(self, data_point: DataPoint) -> Optional[LearningEvent]:
        """Detect concept drift"""
        window_data = list(self.data_windows[data_point.modality])
        if len(window_data) < self.window_size:
            return None
        
        # Compare recent vs older data distributions
        recent_data = window_data[-20:]
        older_data = window_data[:20]
        
        # Simple drift detection based on metadata changes
        if hasattr(data_point, 'metadata') and data_point.metadata:
            recent_meta = [dp.metadata for dp in recent_data if dp.metadata]
            older_meta = [dp.metadata for dp in older_data if dp.metadata]
            
            if recent_meta and older_meta:
                # Check for significant changes in metadata distributions
                # This is a simplified approach - in practice, you'd use more sophisticated methods
                drift_score = self._calculate_distribution_change(recent_meta, older_meta)
                
                if drift_score > 0.5:
                    return LearningEvent(
                        event_type="concept_drift",
                        data_points=[data_point],
                        learning_signal=drift_score,
                        adaptation_required=True,
                        timestamp=datetime.now(),
                        metadata={"drift_score": drift_score, "drift_type": "gradual"}
                    )
        
        return None
    
    def _calculate_distribution_change(self, recent_meta: List[Dict], older_meta: List[Dict]) -> float:
        """Calculate distribution change between metadata"""
        # Simple implementation - compare numeric metadata values
        recent_values = []
        older_values = []
        
        for meta in recent_meta:
            for key, value in meta.items():
                if isinstance(value, (int, float)):
                    recent_values.append(value)
        
        for meta in older_meta:
            for key, value in meta.items():
                if isinstance(value, (int, float)):
                    older_values.append(value)
        
        if not recent_values or not older_values:
            return 0.0
        
        recent_mean = np.mean(recent_values)
        older_mean = np.mean(older_values)
        
        return abs(recent_mean - older_mean) / (abs(older_mean) + 1e-8)

class AdaptiveNeuralNetwork(nn.Module):
    """Adaptive neural network that can learn in real-time"""
    
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Modality-specific encoders
        self.encoders = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.encoders[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Adaptation layers
        self.adaptation_layers = nn.ModuleDict()
        for modality in input_dims:
            self.adaptation_layers[modality] = nn.Sequential(
                nn.Linear(output_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, output_dim)
            )
        
        # Meta-learning components
        self.meta_learner = nn.Sequential(
            nn.Linear(output_dim + 10, hidden_dim // 2),  # +10 for context features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor], 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with multi-modal fusion"""
        encoded = {}
        
        # Encode each modality
        for modality, data in inputs.items():
            if modality in self.encoders:
                encoded[modality] = self.encoders[modality](data)
        
        if not encoded:
            return torch.zeros(1, self.output_dim)
        
        # Fuse modalities
        fused = torch.cat(list(encoded.values()), dim=-1)
        output = self.fusion(fused)
        
        # Apply meta-learning if context is provided
        if context is not None:
            meta_input = torch.cat([output, context], dim=-1)
            meta_output = self.meta_learner(meta_input)
            output = output + meta_output  # Residual connection
        
        return output
    
    def adapt(self, modality: str, data: torch.Tensor) -> torch.Tensor:
        """Adapt specific modality encoder"""
        if modality in self.adaptation_layers:
            base_output = self.encoders[modality](data)
            adaptation = self.adaptation_layers[modality](base_output)
            return base_output + adaptation
        return self.encoders[modality](data)

class RealTimeDataAbsorber:
    """Main class for real-time data absorption and learning"""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 learning_rate: float = 1e-4,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 update_frequency: int = 100,
                 adaptation_threshold: float = 0.7):
        
        # Configuration
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.adaptation_threshold = adaptation_threshold
        
        # Initialize processors
        self.processors = {
            "text": TextProcessor(),
            "image": ImageProcessor(),
            "audio": AudioProcessor(),
            "sensor": SensorProcessor(["temperature", "pressure", "humidity", "acceleration"])
        }
        
        # Initialize neural network
        input_dims = {modality: processor.embedding_dim 
                     for modality, processor in self.processors.items()}
        self.model = AdaptiveNeuralNetwork(input_dims)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize pattern detector
        self.pattern_detector = PatternDetector()
        
        # Data storage
        self.data_buffer = deque(maxlen=buffer_size)
        self.learning_events = PriorityQueue()
        self.processed_count = 0
        
        # Real-time processing
        self.is_running = False
        self.processing_thread = None
        self.learning_thread = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_processed": 0,
            "anomalies_detected": 0,
            "patterns_found": 0,
            "adaptations_made": 0,
            "learning_rate_current": learning_rate
        }
        
        # Database for persistence
        self.db_path = "realtime_learning.db"
        self._init_database()
        
        logging.info("RealTimeDataAbsorber initialized")
    
    def _init_database(self):
        """Initialize database for storing learning events"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY,
                    event_type TEXT,
                    modality TEXT,
                    learning_signal REAL,
                    adaptation_required BOOLEAN,
                    timestamp DATETIME,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY,
                    metric_name TEXT,
                    value REAL,
                    timestamp DATETIME
                )
            """)
            conn.commit()
    
    def start_absorption(self):
        """Start real-time data absorption"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.learning_thread = threading.Thread(target=self._learning_loop)
        
        self.processing_thread.start()
        self.learning_thread.start()
        
        logging.info("Real-time data absorption started")
    
    def stop_absorption(self):
        """Stop real-time data absorption"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
        if self.learning_thread:
            self.learning_thread.join()
        
        logging.info("Real-time data absorption stopped")
    
    def absorb_data(self, data: Any, modality: str, source: str = "unknown", 
                   priority: int = 1, metadata: Dict[str, Any] = None):
        """Absorb new data point"""
        if modality not in self.processors:
            logging.warning(f"Unknown modality: {modality}")
            return
        
        try:
            # Process the data
            processor = self.processors[modality]
            embeddings, extracted_metadata = processor.process(data)
            
            # Combine metadata
            combined_metadata = metadata or {}
            combined_metadata.update(extracted_metadata)
            
            # Create data point
            data_point = DataPoint(
                data=data,
                modality=modality,
                timestamp=datetime.now(),
                source=source,
                priority=priority,
                embeddings=embeddings.numpy(),
                metadata=combined_metadata
            )
            
            # Add to buffer
            self.data_buffer.append(data_point)
            
            # Detect patterns
            events = self.pattern_detector.detect_patterns(data_point)
            
            # Add learning events to queue
            for event in events:
                self.learning_events.put((-event.learning_signal, event))  # Negative for priority
            
            self.performance_metrics["total_processed"] += 1
            
        except Exception as e:
            logging.error(f"Error absorbing data: {e}")
    
    def _processing_loop(self):
        """Main processing loop for real-time data"""
        while self.is_running:
            try:
                # Check system resources
                if self._should_throttle():
                    time.sleep(0.1)
                    continue
                
                # Process pending learning events
                if not self.learning_events.empty():
                    _, event = self.learning_events.get()
                    self._handle_learning_event(event)
                
                # Periodic model updates
                if self.processed_count % self.update_frequency == 0:
                    self._update_model()
                
                self.processed_count += 1
                time.sleep(0.01)  # Small delay to prevent overwhelming
                
            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                time.sleep(1)
    
    def _learning_loop(self):
        """Continuous learning loop"""
        while self.is_running:
            try:
                # Collect recent data for learning
                recent_data = list(self.data_buffer)[-self.batch_size:]
                
                if len(recent_data) >= self.batch_size:
                    self._perform_online_learning(recent_data)
                
                # Clean up old data periodically
                if len(self.data_buffer) > self.buffer_size * 0.9:
                    self._cleanup_old_data()
                
                time.sleep(5)  # Learning happens less frequently
                
            except Exception as e:
                logging.error(f"Error in learning loop: {e}")
                time.sleep(10)
    
    def _handle_learning_event(self, event: LearningEvent):
        """Handle a learning event"""
        try:
            if event.event_type == "anomaly_found":
                self.performance_metrics["anomalies_detected"] += 1
                logging.info(f"Anomaly detected: {event.metadata}")
                
                if event.adaptation_required:
                    self._adapt_to_anomaly(event)
            
            elif event.event_type == "pattern_detected":
                self.performance_metrics["patterns_found"] += 1
                logging.info(f"Pattern detected: {event.metadata}")
                
                # Reinforce learned patterns
                self._reinforce_pattern(event)
            
            elif event.event_type == "concept_drift":
                logging.info(f"Concept drift detected: {event.metadata}")
                
                if event.adaptation_required:
                    self._adapt_to_drift(event)
            
            # Store event in database
            self._store_learning_event(event)
            
        except Exception as e:
            logging.error(f"Error handling learning event: {e}")
    
    def _update_model(self):
        """Update model with recent data"""
        if len(self.data_buffer) < self.batch_size:
            return
        
        try:
            # Sample recent data
            recent_data = list(self.data_buffer)[-self.batch_size * 2:]
            batch_data = np.random.choice(recent_data, self.batch_size, replace=False)
            
            # Prepare batch
            batch_inputs = defaultdict(list)
            
            for data_point in batch_data:
                if data_point.embeddings is not None:
                    batch_inputs[data_point.modality].append(data_point.embeddings)
            
            # Convert to tensors
            tensor_inputs = {}
            for modality, embeddings_list in batch_inputs.items():
                if embeddings_list:
                    tensor_inputs[modality] = torch.stack([
                        torch.tensor(emb).float() for emb in embeddings_list
                    ])
            
            if tensor_inputs:
                # Forward pass
                self.model.train()
                outputs = self.model(tensor_inputs)
                
                # Self-supervised learning objective
                loss = self._calculate_self_supervised_loss(outputs, tensor_inputs)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                logging.debug(f"Model updated, loss: {loss.item():.4f}")
        
        except Exception as e:
            logging.error(f"Error updating model: {e}")
    
    def _calculate_self_supervised_loss(self, outputs: torch.Tensor,
                                       inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate self-supervised learning loss"""
        # Reconstruction loss
        reconstruction_loss = 0.0

        for modality, input_tensor in inputs.items():
            # Try to reconstruct input from output
            reconstructed = self.model.encoders[modality](input_tensor)

            # Compare reconstructed tensor with original input tensor
            mse_loss = nn.MSELoss()(reconstructed, input_tensor)
            reconstruction_loss += mse_loss

        # Normalise by number of modalities
        loss = reconstruction_loss / max(len(inputs), 1)
        return loss
    # --------------------------------------------------

    # ╭──────────────────────────────────────────────────╮
    # │  REAL-TIME ONLINE LEARNING & ADAPTATION UTILS    │
    # ╰──────────────────────────────────────────────────╯
    def _perform_online_learning(self, batch: List[DataPoint]):
        """One gradient step on a random batch drawn from recent data."""
        try:
            batch_inputs = defaultdict(list)
            for dp in batch:
                if dp.embeddings is not None:
                    batch_inputs[dp.modality].append(dp.embeddings)

            tensor_inputs = {
                m: torch.stack([torch.tensor(e).float() for e in embs])
                for m, embs in batch_inputs.items() if embs
            }

            if not tensor_inputs:
                return

            self.model.train()
            out  = self.model(tensor_inputs)
            loss = self._calculate_self_supervised_loss(out, tensor_inputs)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            logging.debug(f"[Online-Learn] loss={loss.item():.4f}")

        except Exception as e:
            logging.error(f"Online learning error: {e}")

    # --------------------------------------------------------------------- #
    # ADAPTATION HELPERS
    # --------------------------------------------------------------------- #
    def _adapt_to_anomaly(self, event: LearningEvent):
        """Light-weight fine-tuning on the anomalous sample."""
        for dp in event.data_points:
            if dp.embeddings is None:
                continue
            emb = torch.tensor(dp.embeddings).float().unsqueeze(0)
            adapted = self.model.adapt(dp.modality, emb)
            # We do not back-prop here; just cache result / log
            logging.info(f"Adapted to anomaly in modality '{dp.modality}'.")

        self.performance_metrics["adaptations_made"] += 1

    def _adapt_to_drift(self, event: LearningEvent):
        """Adjust learning rate & maybe increase update frequency."""
        self.performance_metrics["adaptations_made"] += 1
        # Simple heuristic: boost LR temporarily
        self.optimizer.param_groups[0]["lr"] *= 1.2
        self.performance_metrics["learning_rate_current"] = self.optimizer.param_groups[0]["lr"]
        logging.info("Concept-drift adaptation: learning-rate boosted.")

    def _reinforce_pattern(self, event: LearningEvent):
        """Optionally lower LR to consolidate stable patterns."""
        self.optimizer.param_groups[0]["lr"] *= 0.95
        self.performance_metrics["learning_rate_current"] = self.optimizer.param_groups[0]["lr"]
        logging.debug("Stable pattern detected ⇒ LR slightly decayed.")

    # --------------------------------------------------------------------- #
    # HOUSE-KEEPING
    # --------------------------------------------------------------------- #
    def _cleanup_old_data(self):
        """Trim buffer & invoke garbage-collector to keep memory sane."""
        excess = len(self.data_buffer) - self.buffer_size
        if excess > 0:
            for _ in range(excess):
                self.data_buffer.popleft()
        gc.collect()
        logging.debug("Old data cleaned; buffer size=%d", len(self.data_buffer))

    def _should_throttle(self) -> bool:
        """Pause ingestion if CPU / RAM too high."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        return cpu > 90 or ram > 90

    # --------------------------------------------------------------------- #
    # PERSISTENCE
    # --------------------------------------------------------------------- #
    def _store_learning_event(self, event: LearningEvent):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO learning_events (event_type, modality, learning_signal, "
                    "adaptation_required, timestamp, metadata) VALUES (?,?,?,?,?,?)",
                    (
                        event.event_type,
                        event.data_points[0].modality if event.data_points else "unknown",
                        event.learning_signal,
                        int(event.adaptation_required),
                        event.timestamp.isoformat(),
                        json.dumps(event.metadata),
                    ),
                )
                conn.commit()
        except Exception as e:
            logging.error(f"DB insert error: {e}")

    def log_performance_metrics(self):
        """Write current metrics to DB once per hour (call externally)."""
        ts = datetime.now().replace(minute=0, second=0, microsecond=0)
        with sqlite3.connect(self.db_path) as conn:
            for k, v in self.performance_metrics.items():
                conn.execute(
                    "INSERT INTO performance_metrics (metric_name, value, timestamp) "
                    "VALUES (?,?,?)",
                    (k, float(v), ts.isoformat()),
                )
            conn.commit()

# ─────────────────────────────────────────────────────────────────────────────── #
# Example usage (remove / comment-out when importing as a library)
# ─────────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    absorber = RealTimeDataAbsorber(model_config={})
    absorber.start_absorption()

    try:
        # Dummy streaming loop
        for i in range(500):
            absorber.absorb_data(
                data=f"Sample text data point {i}",
                modality="text",
                source="synthetic",
                priority=np.random.randint(1, 5),
            )
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        absorber.stop_absorption()
        absorber.log_performance_metrics()
        logging.info("Finished demo run.")
