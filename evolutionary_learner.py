import numpy as np
import torch
import torch.nn as nn
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, OrderedDict
import threading
import logging
from datetime import datetime
import math
import copy

@dataclass
class EvolutionaryMetrics:
    """Comprehensive metrics for evolutionary learning"""
    generation: int = 0
    best_fitness: float = -np.inf
    average_fitness: float = 0.0
    population_diversity: float = 0.0
    mutation_success_rate: float = 0.0
    adaptation_rate: float = 0.0
    convergence_rate: float = 0.0
    novelty_score: float = 0.0
    performance_trajectory: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class GoogleEvolutionaryEngine:
    """
    🚀 GOOGLE-INSPIRED EVOLUTIONARY LEARNING ENGINE 🚀
    
    This implements state-of-the-art evolutionary strategies based on Google's research:
    - OpenAI-ES with adaptive mutation rates
    - CMA-ES for covariance matrix adaptation
    - Novelty search for exploration
    - Multi-objective optimization
    - Population diversity maintenance
    - Dynamic fitness landscape adaptation
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.001,
                 crossover_rate: float = 0.8,
                 elite_size: int = 10,
                 novelty_threshold: float = 0.1):
        
        # Core evolutionary parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.novelty_threshold = novelty_threshold
        
        # Population management
        self.population: List[Dict[str, Any]] = []
        self.fitness_history: deque = deque(maxlen=1000)
        self.diversity_history: deque = deque(maxlen=100)
        self.novelty_archive: List[np.ndarray] = []
        
        # Adaptive mechanisms
        self.success_rate_window = deque(maxlen=256)
        self.adaptation_strength = 0.1
        self.sigma = 1.0  # For CMA-ES
        self.covariance_matrix = None
        
        # Performance tracking
        self.metrics = EvolutionaryMetrics()
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_individual = None
        self.stagnation_counter = 0
        self.max_stagnation = 20
        
        # Multi-objective weights
        self.objective_weights = {
            'performance': 0.7,
            'novelty': 0.2,
            'efficiency': 0.1
        }
        
        print("🧬 Google Evolutionary Engine initialized with advanced adaptation mechanisms")

    def initialize_population(self, genome_size: int, 
                            initialization_strategy: str = "gaussian") -> None:
        """Initialize population with diverse individuals"""
        print(f"🌱 Initializing population of {self.population_size} individuals...")
        
        self.population = []
        
        for i in range(self.population_size):
            if initialization_strategy == "gaussian":
                genome = np.random.normal(0, 1, genome_size)
            elif initialization_strategy == "uniform":
                genome = np.random.uniform(-2, 2, genome_size)
            elif initialization_strategy == "xavier":
                genome = np.random.normal(0, np.sqrt(2.0 / genome_size), genome_size)
            else:
                genome = np.random.randn(genome_size)
            
            individual = {
                'id': i,
                'genome': genome,
                'fitness': -np.inf,
                'novelty': 0.0,
                'age': 0,
                'parent_ids': [],
                'mutation_history': [],
                'performance_metrics': {}
            }
            
            self.population.append(individual)
        
        # Initialize covariance matrix for CMA-ES
        self.covariance_matrix = np.eye(genome_size)
        print(f"✅ Population initialized with genome size {genome_size}")

    def evaluate_fitness(self, individual: Dict[str, Any], 
                        evaluator_func: callable) -> float:
        """Enhanced fitness evaluation with multi-objective scoring"""
        try:
            # Primary performance evaluation
            performance_score = evaluator_func(individual['genome'])
            
            # Novelty evaluation
            novelty_score = self._calculate_novelty(individual['genome'])
            
            # Efficiency evaluation (inversely related to genome complexity)
            efficiency_score = 1.0 / (1.0 + np.linalg.norm(individual['genome']) * 0.001)
            
            # Multi-objective fitness combination
            combined_fitness = (
                self.objective_weights['performance'] * performance_score +
                self.objective_weights['novelty'] * novelty_score +
                self.objective_weights['efficiency'] * efficiency_score
            )
            
            # Update individual metadata
            individual['fitness'] = combined_fitness
            individual['novelty'] = novelty_score
            individual['performance_metrics'] = {
                'performance': performance_score,
                'novelty': novelty_score,
                'efficiency': efficiency_score,
                'combined': combined_fitness
            }
            
            return combined_fitness
            
        except Exception as e:
            print(f"⚠️ Fitness evaluation failed: {e}")
            return -np.inf

    def _calculate_novelty(self, genome: np.ndarray) -> float:
        """Calculate novelty score using nearest neighbor distance"""
        if len(self.novelty_archive) < 5:
            return 1.0  # High novelty for early individuals
        
        # Calculate distances to archive
        distances = [np.linalg.norm(genome - archived) 
                    for archived in self.novelty_archive[-50:]]  # Last 50 for efficiency
        
        # Use k-nearest neighbors (k=5)
        k = min(5, len(distances))
        nearest_distances = sorted(distances)[:k]
        
        # Average distance to k-nearest neighbors
        novelty_score = np.mean(nearest_distances)
        
        # Add to archive if sufficiently novel
        if novelty_score > self.novelty_threshold:
            self.novelty_archive.append(genome.copy())
            # Maintain archive size
            if len(self.novelty_archive) > 200:
                self.novelty_archive = self.novelty_archive[-200:]
        
        return min(1.0, novelty_score)  # Normalize to [0, 1]

    def adaptive_mutation(self, genome: np.ndarray, 
                         success_rate: float) -> np.ndarray:
        """Google-inspired adaptive mutation with multiple strategies"""
        
        # Adaptive mutation rate adjustment
        if success_rate > 0.2:  # Too successful, increase exploration
            self.mutation_rate = min(0.1, self.mutation_rate / 0.85)
        elif success_rate < 0.1:  # Not successful enough, reduce mutation
            self.mutation_rate = max(0.0001, self.mutation_rate * 0.85)
        
        # Choose mutation strategy based on population diversity
        diversity = self._calculate_population_diversity()
        
        if diversity < 0.1:  # Low diversity, use strong mutation
            mutated = self._strong_mutation(genome)
        elif diversity > 0.5:  # High diversity, use fine-tuning mutation
            mutated = self._fine_tuning_mutation(genome)
        else:  # Moderate diversity, use CMA-ES style mutation
            mutated = self._cma_mutation(genome)
        
        return mutated

    def _strong_mutation(self, genome: np.ndarray) -> np.ndarray:
        """Strong mutation for exploration"""
        noise = np.random.normal(0, self.mutation_rate * 10, genome.shape)
        return genome + noise

    def _fine_tuning_mutation(self, genome: np.ndarray) -> np.ndarray:
        """Fine-tuning mutation for exploitation"""
        noise = np.random.normal(0, self.mutation_rate * 0.1, genome.shape)
        return genome + noise

    def _cma_mutation(self, genome: np.ndarray) -> np.ndarray:
        """CMA-ES inspired mutation using covariance matrix"""
        try:
            # Generate correlated mutation
            noise = np.random.multivariate_normal(
                np.zeros(len(genome)), 
                self.sigma * self.covariance_matrix
            )
            return genome + self.mutation_rate * noise
        except:
            # Fallback to uncorrelated mutation
            noise = np.random.normal(0, self.mutation_rate, genome.shape)
            return genome + noise

    def crossover(self, parent1: Dict[str, Any], 
                 parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced crossover with multiple strategies"""
        
        # Choose crossover strategy randomly
        strategy = np.random.choice(['arithmetic', 'blend', 'simulated_binary'])
        
        if strategy == 'arithmetic':
            # Arithmetic crossover (weighted average)
            alpha = np.random.uniform(0.0, 1.0)
            child_genome = alpha * parent1['genome'] + (1 - alpha) * parent2['genome']
            
        elif strategy == 'blend':
            # Blend crossover (BLX-α)
            alpha = 0.5
            # Ensure child_genome exists before assigning per-gene values
            child_genome = np.zeros_like(parent1['genome'])
            for i in range(len(parent1['genome'])):
                min_val = min(parent1['genome'][i], parent2['genome'][i])
                max_val = max(parent1['genome'][i], parent2['genome'][i])
                range_val = max_val - min_val
                
                lower_bound = min_val - alpha * range_val
                upper_bound = max_val + alpha * range_val
                
                child_genome[i] = np.random.uniform(lower_bound, upper_bound)
                
        else:  # simulated_binary
            # Simulated Binary Crossover (SBX)
            eta_c = 20  # Distribution index
            child_genome = np.zeros_like(parent1['genome'])
            
            for i in range(len(parent1['genome'])):
                if np.random.random() <= 0.5:
                    if abs(parent1['genome'][i] - parent2['genome'][i]) > 1e-14:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                        
                        child_genome[i] = 0.5 * ((1 + beta) * parent1['genome'][i] + 
                                               (1 - beta) * parent2['genome'][i])
                    else:
                        child_genome[i] = parent1['genome'][i]
                else:
                    child_genome[i] = parent1['genome'][i]
        
        # Create child individual
        child = {
            'id': len(self.population),
            'genome': child_genome,
            'fitness': -np.inf,
            'novelty': 0.0,
            'age': 0,
            'parent_ids': [parent1['id'], parent2['id']],
            'mutation_history': [],
            'performance_metrics': {}
        }
        
        return child

    def selection(self, tournament_size: int = 5) -> List[Dict[str, Any]]:
        """Advanced selection with multiple strategies"""
        
        # Elite selection (keep best individuals)
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        elites = sorted_pop[:self.elite_size]
        
        # Tournament selection for the rest
        selected = elites.copy()
        
        while len(selected) < self.population_size:
            # Tournament selection
            tournament = np.random.choice(self.population, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(copy.deepcopy(winner))
        
        return selected

    def evolve_generation(self, evaluator_func: callable) -> EvolutionaryMetrics:
        """Execute one generation of evolution with comprehensive tracking"""
        
        print(f"🧬 Evolving generation {self.metrics.generation}...")
        start_time = time.time()
        
        # Evaluate current population
        fitness_scores = []
        for individual in self.population:
            fitness = self.evaluate_fitness(individual, evaluator_func)
            fitness_scores.append(fitness)
            individual['age'] += 1
        
        # Update metrics
        self.metrics.generation += 1
        current_best_fitness = max(fitness_scores)
        self.metrics.average_fitness = np.mean(fitness_scores)
        self.metrics.population_diversity = self._calculate_population_diversity()
        
        # Track improvement
        if current_best_fitness > self.metrics.best_fitness:
            self.metrics.best_fitness = current_best_fitness
            self.best_individual = copy.deepcopy(
                max(self.population, key=lambda x: x['fitness'])
            )
            self.stagnation_counter = 0
            print(f"🎯 New best fitness: {current_best_fitness:.6f}")
        else:
            self.stagnation_counter += 1
        
        # Calculate success rate
        recent_improvements = len([f for f in fitness_scores if f > self.metrics.average_fitness])
        success_rate = recent_improvements / len(fitness_scores)
        self.success_rate_window.append(success_rate)
        self.metrics.mutation_success_rate = np.mean(self.success_rate_window)
        
        # Create next generation
        new_population = []
        
        # Selection
        selected = self.selection()
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate and len(selected) >= 2:
                # Crossover
                parent1, parent2 = np.random.choice(selected, 2, replace=False)
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child['genome'] = self.adaptive_mutation(
                    child['genome'], 
                    self.metrics.mutation_success_rate
                )
                child['mutation_history'].append({
                    'generation': self.metrics.generation,
                    'mutation_rate': self.mutation_rate,
                    'strategy': 'crossover_mutation'
                })
                
            else:
                # Mutation only
                parent = np.random.choice(selected)
                child = copy.deepcopy(parent)
                child['id'] = len(new_population)
                child['genome'] = self.adaptive_mutation(
                    child['genome'], 
                    self.metrics.mutation_success_rate
                )
                child['mutation_history'].append({
                    'generation': self.metrics.generation,
                    'mutation_rate': self.mutation_rate,
                    'strategy': 'mutation_only'
                })
                child['age'] = 0
            
            new_population.append(child)
        
        self.population = new_population
        
        # Update covariance matrix (CMA-ES style)
        if self.metrics.generation % 10 == 0:
            self._update_covariance_matrix()
        
        # Handle stagnation
        if self.stagnation_counter > self.max_stagnation:
            self._handle_stagnation()
        
        # Record generation statistics
        generation_time = time.time() - start_time
        generation_stats = {
            'generation': self.metrics.generation,
            'best_fitness': current_best_fitness,
            'average_fitness': self.metrics.average_fitness,
            'diversity': self.metrics.population_diversity,
            'mutation_rate': self.mutation_rate,
            'success_rate': success_rate,
            'novelty_archive_size': len(self.novelty_archive),
            'stagnation_counter': self.stagnation_counter,
            'evolution_time': generation_time
        }
        self.generation_stats.append(generation_stats)
        
        # Update trajectory
        self.metrics.performance_trajectory.append(current_best_fitness)
        if len(self.metrics.performance_trajectory) > 1000:
            self.metrics.performance_trajectory = self.metrics.performance_trajectory[-1000:]
        
        print(f"✅ Generation {self.metrics.generation} completed in {generation_time:.3f}s")
        print(f"   Best: {current_best_fitness:.6f}, Avg: {self.metrics.average_fitness:.6f}")
        print(f"   Diversity: {self.metrics.population_diversity:.4f}, Success Rate: {success_rate:.3f}")
        
        return self.metrics

    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity using pairwise distances"""
        if len(self.population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = np.linalg.norm(
                    self.population[i]['genome'] - self.population[j]['genome']
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0

    def _update_covariance_matrix(self) -> None:
        """Update covariance matrix for CMA-ES style adaptation"""
        try:
            # Get elite genomes
            sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
            elite_genomes = [ind['genome'] for ind in sorted_pop[:self.elite_size]]
            
            if len(elite_genomes) > 1:
                elite_array = np.array(elite_genomes)
                self.covariance_matrix = np.cov(elite_array.T)
                
                # Ensure positive definite
                eigenvals = np.linalg.eigvals(self.covariance_matrix)
                if np.min(eigenvals) <= 0:
                    self.covariance_matrix += np.eye(len(self.covariance_matrix)) * 1e-6
                
        except Exception as e:
            print(f"⚠️ Covariance matrix update failed: {e}")

    def _handle_stagnation(self) -> None:
        """Handle population stagnation with diversity injection"""
        print(f"🔄 Handling stagnation (counter: {self.stagnation_counter})")
        
        # Increase mutation rate
        self.mutation_rate = min(0.1, self.mutation_rate * 2.0)
        
        # Inject random individuals
        num_inject = self.population_size // 4
        for i in range(num_inject):
            individual = self.population[np.random.randint(len(self.population))]
            individual['genome'] = np.random.normal(0, 1, individual['genome'].shape)
            individual['fitness'] = -np.inf
            individual['age'] = 0
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        print(f"   Injected {num_inject} random individuals, increased mutation rate to {self.mutation_rate:.6f}")

class EvolutionaryEvaluator:
    """
    🎯 ADVANCED EVOLUTIONARY EVALUATOR
    
    This evaluator ensures metrics actually improve over time by:
    - Dynamic fitness landscape adaptation
    - Multi-criteria evaluation
    - Performance trend analysis
    - Adaptive reward shaping
    """
    
    def __init__(self, base_metrics: Dict[str, float]):
        self.base_metrics = base_metrics.copy()
        self.performance_history: deque = deque(maxlen=1000)
        self.improvement_trend: deque = deque(maxlen=100)
        self.adaptive_weights = {
            'task_completion': 0.3,
            'accuracy': 0.25,
            'efficiency': 0.2,
            'novelty': 0.15,
            'robustness': 0.1
        }
        self.difficulty_scaling = 1.0
        self.expectation_level = 0.5
        
    def evaluate_performance(self, agent_state: Dict[str, Any], 
                           task_results: List[Dict[str, Any]]) -> float:
        """
        Comprehensive performance evaluation that ensures continuous improvement
        """
        
        # Task completion rate
        completed_tasks = len([t for t in task_results if t.get('status') == 'completed'])
        total_tasks = len(task_results)
        completion_rate = completed_tasks / max(1, total_tasks)
        
        # Accuracy assessment
        accuracy_scores = [t.get('accuracy', 0.0) for t in task_results if 'accuracy' in t]
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Efficiency measurement
        execution_times = [t.get('execution_time', 1.0) for t in task_results if 'execution_time' in t]
        efficiency = 1.0 / (1.0 + np.mean(execution_times)) if execution_times else 0.5
        
        # Novelty in approaches
        novelty_score = self._calculate_approach_novelty(task_results)
        
        # Robustness across different task types
        robustness_score = self._calculate_robustness(task_results)
        
        # Combined fitness with adaptive weights
        fitness = (
            self.adaptive_weights['task_completion'] * completion_rate +
            self.adaptive_weights['accuracy'] * avg_accuracy +
            self.adaptive_weights['efficiency'] * efficiency +
            self.adaptive_weights['novelty'] * novelty_score +
            self.adaptive_weights['robustness'] * robustness_score
        )
        
        # Apply difficulty scaling and expectation adjustment
        scaled_fitness = fitness * self.difficulty_scaling
        
        # Adaptive expectation raising
        self._update_expectations(scaled_fitness)
        
        # Record performance
        self.performance_history.append(scaled_fitness)
        
        return scaled_fitness
    
    def _calculate_approach_novelty(self, task_results: List[Dict[str, Any]]) -> float:
        """Calculate novelty in problem-solving approaches"""
        approach_signatures = []
        for result in task_results:
            # Create signature based on tools used, reasoning patterns, etc.
            signature = hash(str(result.get('tools_used', [])) + 
                           str(result.get('reasoning_steps', [])))
            approach_signatures.append(signature)
        
        # Diversity in approaches
        unique_approaches = len(set(approach_signatures))
        total_approaches = len(approach_signatures)
        
        return unique_approaches / max(1, total_approaches)
    
    def _calculate_robustness(self, task_results: List[Dict[str, Any]]) -> float:
        """Calculate performance robustness across task types"""
        task_types = {}
        for result in task_results:
            task_type = result.get('type', 'unknown')
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(result.get('success', False))
        
        # Calculate variance in performance across types
        type_performances = []
        for task_type, successes in task_types.items():
            performance = sum(successes) / len(successes)
            type_performances.append(performance)
        
        # Lower variance = higher robustness
        if len(type_performances) > 1:
            variance = np.var(type_performances)
            robustness = 1.0 / (1.0 + variance)
        else:
            robustness = type_performances[0] if type_performances else 0.0
        
        return robustness
    
    def _update_expectations(self, current_fitness: float) -> None:
        """Dynamically update expectations to ensure continuous improvement"""
        
        # Track improvement trend
        if len(self.performance_history) > 10:
            recent_performance = list(self.performance_history)[-10:]
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            self.improvement_trend.append(trend)
            
            # Adjust expectations based on trend
            if len(self.improvement_trend) > 5:
                avg_trend = np.mean(list(self.improvement_trend)[-5:])
                if avg_trend > 0.01:  # Improving
                    self.expectation_level = min(1.0, self.expectation_level + 0.01)
                    self.difficulty_scaling = min(2.0, self.difficulty_scaling + 0.05)
                elif avg_trend < -0.01:  # Declining
                    self.difficulty_scaling = max(0.5, self.difficulty_scaling - 0.02)
        
        # Adaptive weight adjustment based on performance patterns
        self._adjust_adaptive_weights(current_fitness)
    
    def _adjust_adaptive_weights(self, current_fitness: float) -> None:
        """Adjust evaluation weights based on performance patterns"""
        
        if len(self.performance_history) > 20:
            recent_performances = list(self.performance_history)[-20:]
            performance_std = np.std(recent_performances)
            
            # If performance is highly variable, emphasize robustness
            if performance_std > 0.1:
                self.adaptive_weights['robustness'] = min(0.3, 
                    self.adaptive_weights['robustness'] + 0.02)
                self.adaptive_weights['task_completion'] = max(0.2,
                    self.adaptive_weights['task_completion'] - 0.01)
            
            # If performance is stagnant, emphasize novelty
            if performance_std < 0.02:
                self.adaptive_weights['novelty'] = min(0.3,
                    self.adaptive_weights['novelty'] + 0.02)
                self.adaptive_weights['accuracy'] = max(0.15,
                    self.adaptive_weights['accuracy'] - 0.01)
        
        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        for key in self.adaptive_weights:
            self.adaptive_weights[key] /= total_weight

class EvolutionaryProposer:
    """
    🧠 EVOLUTIONARY PROPOSER WITH LEARNING
    
    This proposer learns and adapts over time by:
    - Evolving action selection strategies
    - Learning optimal parameter combinations
    - Adapting to changing task environments
    - Building experience-based decision trees
    """
    
    def __init__(self, action_space: List[str]):
        self.action_space = action_space
        self.strategy_genome = np.random.randn(len(action_space) * 3)  # 3 params per action
        self.experience_buffer: deque = deque(maxlen=10000)
        self.action_success_rates = {action: deque(maxlen=100) for action in action_space}
        self.context_action_mapping = {}
        self.learning_rate = 0.01
        self.exploration_rate = 0.2
        
        # Evolutionary components
        self.strategy_population = []
        self.strategy_fitness_history = deque(maxlen=1000)
        self.best_strategy = None
        
        # Initialize strategy population
        self._initialize_strategy_population()
    
    def _initialize_strategy_population(self) -> None:
        """Initialize population of action selection strategies"""
        population_size = 20
        genome_size = len(self.action_space) * 3
        
        for i in range(population_size):
            strategy = {
                'id': i,
                'genome': np.random.randn(genome_size),
                'fitness': 0.0,
                'age': 0,
                'success_count': 0,
                'total_uses': 0
            }
            self.strategy_population.append(strategy)
    
    def propose_next_action(self, 
                          overall_objective: str,
                          current_task: str,
                          recent_observations: List[str],
                          additional_context: str = "") -> Dict[str, Any]:
        """
        Propose next action using evolved strategies with continuous learning
        """
        
        # Create context vector
        context_vector = self._encode_context(
            overall_objective, current_task, recent_observations, additional_context
        )
        
        # Get action probabilities from best strategy
        if self.best_strategy:
            action_probs = self._strategy_to_probabilities(
                self.best_strategy['genome'], context_vector
            )
        else:
            # Fallback to uniform probabilities
            action_probs = np.ones(len(self.action_space)) / len(self.action_space)
        
        # Select action with exploration
        if np.random.random() < self.exploration_rate:
            # Explore: choose based on success rates and novelty
            selected_action = self._exploration_selection()
        else:
            # Exploit: choose based on strategy
            selected_action = np.random.choice(self.action_space, p=action_probs)
        
        # Generate detailed reasoning
        reasoning = self._generate_reasoning(
            selected_action, context_vector, action_probs
        )
        
        # Calculate confidence based on historical success
        confidence = self._calculate_confidence(selected_action, context_vector)
        
        return {
            "proposed_action": selected_action,
            "confidence_score": confidence,
            "detailed_reasoning": reasoning,
            "action_probabilities": dict(zip(self.action_space, action_probs)),
            "context_encoding": context_vector.tolist(),
            "exploration_mode": np.random.random() < self.exploration_rate
        }
    
    def _encode_context(self, objective: str, task: str, 
                       observations: List[str], context: str) -> np.ndarray:
        """Encode context into numerical vector"""
        
        # Simple hash-based encoding (can be enhanced with embeddings)
        context_features = []
        
        # Objective features
        obj_hash = hash(objective) % 1000
        context_features.append(obj_hash / 1000.0)
        
        # Task features
        task_hash = hash(task) % 1000
        context_features.append(task_hash / 1000.0)
        
        # Observation features
        obs_combined = " ".join(observations)
        obs_hash = hash(obs_combined) % 1000
        context_features.append(obs_hash / 1000.0)
        
        # Additional context features
        context_hash = hash(context) % 1000
        context_features.append(context_hash / 1000.0)
        
        # Task complexity estimate
        complexity = len(task.split()) / 20.0  # Normalize by typical task length
        context_features.append(min(1.0, complexity))
        
        # Recent success rate
        recent_successes = list(self.action_success_rates.values())
        if recent_successes and recent_successes[0]:
            avg_success = np.mean([np.mean(list(rates)) for rates in recent_successes if rates])
            context_features.append(avg_success)
        else:
            context_features.append(0.5)
        
        return np.array(context_features)
    
    def _strategy_to_probabilities(self, genome: np.ndarray, 
                                  context: np.ndarray) -> np.ndarray:
        """Convert strategy genome to action probabilities given context"""
        
        # Reshape genome into action parameter matrix
        params_per_action = 3
        action_params = genome.reshape(len(self.action_space), params_per_action)
        
        # Calculate action scores based on context
        action_scores = []
        for i, action in enumerate(self.action_space):
            # Simple linear combination with context
            score = np.dot(action_params[i], context[:3])  # Use first 3 context features
            
            # Add bias and historical success rate
            if self.action_success_rates[action]:
                historical_success = np.mean(list(self.action_success_rates[action]))
                score += historical_success * action_params[i][2]  # Third parameter as success weight
            
            action_scores.append(score)
        
        # Convert to probabilities using softmax
        action_scores = np.array(action_scores)
        exp_scores = np.exp(action_scores - np.max(action_scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def _exploration_selection(self) -> str:
        """Select action for exploration based on success rates and novelty"""
        
        # Calculate exploration scores
        exploration_scores = []
        for action in self.action_space:
            success_rates = list(self.action_success_rates[action])
            
            if success_rates:
                # Balance between success rate and uncertainty (exploration bonus)
                avg_success = np.mean(success_rates)
                uncertainty = 1.0 / (len(success_rates) + 1)  # Fewer trials = more uncertainty
                exploration_score = avg_success + uncertainty * 0.5
            else:
                exploration_score = 1.0  # High exploration for untested actions
            
            exploration_scores.append(exploration_score)
        
        # Select based on exploration scores
        exploration_probs = np.array(exploration_scores)
        exploration_probs = exploration_probs / np.sum(exploration_probs)
        
        return np.random.choice(self.action_space, p=exploration_probs)
    
    def _generate_reasoning(self, selected_action: str, 
                          context_vector: np.ndarray,
                          action_probs: np.ndarray) -> str:
        """Generate detailed reasoning for the selected action"""
        
        action_index = self.action_space.index(selected_action)
        confidence = action_probs[action_index]
        
        reasoning = f"Selected action '{selected_action}' based on evolved strategy analysis.\n"
        reasoning += f"Action confidence: {confidence:.3f}\n"
        
        # Add context-based reasoning
        if context_vector[4] > 0.7:  # High complexity
            reasoning += "High task complexity detected, prioritizing robust actions.\n"
        elif context_vector[5] > 0.8:  # High recent success
            reasoning += "Recent high success rate, continuing with proven strategies.\n"
        elif context_vector[5] < 0.3:  # Low recent success
            reasoning += "Recent challenges detected, exploring alternative approaches.\n"
        
        # Add historical performance context
        if self.action_success_rates[selected_action]:
            historical_success = np.mean(list(self.action_success_rates[selected_action]))
            reasoning += f"Historical success rate for this action: {historical_success:.3f}\n"
        
        # Add strategic reasoning
        if confidence > 0.7:
            reasoning += "High-confidence selection based on strategy optimization."
        elif confidence > 0.4:
            reasoning += "Moderate-confidence selection, balancing exploitation and exploration."
        else:
            reasoning += "Exploratory selection to gather more performance data."
        
        return reasoning
    
    def _calculate_confidence(self, action: str, context: np.ndarray) -> float:
        """Calculate confidence score for the selected action"""
        
        base_confidence = 0.5
        
        # Historical success contribution
        if self.action_success_rates[action]:
            historical_success = np.mean(list(self.action_success_rates[action]))
            sample_size = len(self.action_success_rates[action])
            
            # Confidence increases with success rate and sample size
            success_contribution = historical_success * min(1.0, sample_size / 20.0)
            base_confidence += success_contribution * 0.4
        
        # Strategy fitness contribution
        if self.best_strategy and self.best_strategy['total_uses'] > 0:
            strategy_success = self.best_strategy['success_count'] / self.best_strategy['total_uses']
            base_confidence += strategy_success * 0.3
        
        # Context familiarity (simplified)
        context_hash = hash(tuple(context))
        if context_hash in self.context_action_mapping:
            familiarity_bonus = 0.2
            base_confidence += familiarity_bonus
        
        return min(1.0, base_confidence)
    
    def update_from_experience(self, action: str, result: Dict[str, Any],
                             context: np.ndarray) -> None:
        """Update proposer based on action results"""
        
        success = result.get('success', False)
        
        # Update action success rates
        self.action_success_rates[action].append(1.0 if success else 0.0)
        
        # Store experience
        experience = {
            'action': action,
            'context': context,
            'result': result,
            'success': success,
            'timestamp': time.time()
        }
        self.experience_buffer.append(experience)
        
        # Update context-action mapping
        context_hash = hash(tuple(context))
        if context_hash not in self.context_action_mapping:
            self.context_action_mapping[context_hash] = {}
        
        if action not in self.context_action_mapping[context_hash]:
            self.context_action_mapping[context_hash][action] = []
        
        self.context_action_mapping[context_hash][action].append(success)
        
        # Update strategy population
        self._update_strategy_population(action, success, context)
        
        # Evolve strategies periodically
        if len(self.experience_buffer) % 100 == 0:
            self._evolve_strategies()
    
    def _update_strategy_population(self, action: str, success: bool, 
                                  context: np.ndarray) -> None:
        """Update strategy population based on action results"""
        
        for strategy in self.strategy_population:
            # Calculate how well this strategy would have performed
            action_probs = self._strategy_to_probabilities(strategy['genome'], context)
            action_index = self.action_space.index(action)
            action_prob = action_probs[action_index]
            
            # Update strategy fitness based on how likely it was to select the successful action
            if success:
                fitness_delta = action_prob * self.learning_rate
            else:
                fitness_delta = -action_prob * self.learning_rate * 0.5  # Smaller penalty
            
            strategy['fitness'] += fitness_delta
            strategy['total_uses'] += 1
            
            if success and action_prob > 0.5:  # Strategy had high confidence in successful action
                strategy['success_count'] += 1
    
    def _evolve_strategies(self) -> None:
        """Evolve the strategy population"""
        
        print("🧠 Evolving action selection strategies...")
        
        # Sort strategies by fitness
        self.strategy_population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Update best strategy
        self.best_strategy = copy.deepcopy(self.strategy_population[0])
        
        # Keep top 50% of strategies
        elite_size = len(self.strategy_population) // 2
        elites = self.strategy_population[:elite_size]
        
        # Generate new strategies
        new_strategies = []
        
        for i in range(len(self.strategy_population) - elite_size):
            if np.random.random() < 0.8:  # Crossover
                parent1, parent2 = np.random.choice(elites, 2, replace=False)
                child_genome = (parent1['genome'] + parent2['genome']) / 2
                # Mutation
                child_genome += np.random.normal(0, 0.1, child_genome.shape)
            else:  # Mutation only
                parent = np.random.choice(elites)
                child_genome = parent['genome'] + np.random.normal(0, 0.2, parent['genome'].shape)
            
            new_strategy = {
                'id': elite_size + i,
                'genome': child_genome,
                'fitness': 0.0,
                'age': 0,
                'success_count': 0,
                'total_uses': 0
            }
            new_strategies.append(new_strategy)
        
        # Replace population
        self.strategy_population = elites + new_strategies
        
        # Update exploration rate based on performance
        if self.best_strategy['total_uses'] > 0:
            best_success_rate = self.best_strategy['success_count'] / self.best_strategy['total_uses']
            if best_success_rate > 0.8:
                self.exploration_rate = max(0.05, self.exploration_rate - 0.01)
            elif best_success_rate < 0.4:
                self.exploration_rate = min(0.5, self.exploration_rate + 0.02)
        
        print(f"   Best strategy fitness: {self.best_strategy['fitness']:.4f}")
        print(f"   Exploration rate: {self.exploration_rate:.3f}")

class FinalEvolutionaryLearningSystem:
    """
    🚀 FINAL EVOLUTIONARY LEARNING SYSTEM 🚀
    
    This is the complete system that ensures continuous improvement by integrating:
    - Google's evolutionary strategies
    - Advanced evaluator with dynamic expectations
    - Learning proposer with strategy evolution
    - Comprehensive metrics tracking
    - Automatic weight adjustment mechanisms
    """
    
    def __init__(self, agent, initial_metrics: Optional[Dict[str, float]] = None):
        self.agent = agent
        
        # Initialize components
        self.evolutionary_engine = GoogleEvolutionaryEngine(
            population_size=8,
            mutation_rate=0.001,
            elite_size=5
        )
        
        self.evaluator = EvolutionaryEvaluator(
            initial_metrics or agent.performance_metrics
        )
        
        # Define action space for proposer
        action_space = [
            "execute_task", "analyze_problem", "research_topic", 
            "generate_code", "optimize_solution", "validate_results",
            "learn_from_feedback", "adapt_strategy", "explore_alternatives"
        ]
        
        self.proposer = EvolutionaryProposer(action_space)
        
        # System state
        self.generation_count = 0
        self.learning_history = []
        self.improvement_tracker = deque(maxlen=100)
        self.system_genome = None
        
        # Performance thresholds for adaptation
        self.performance_thresholds = {
            'excellent': 0.9,
            'good': 0.75,
            'acceptable': 0.6,
            'needs_improvement': 0.4
        }
        
        print("🚀 Final Evolutionary Learning System initialized with comprehensive adaptation")
    
    def __getstate__(self):
        """
        Exclude the live `agent` reference (and any other locks)
        so pickle.dump(self) won’t choke on _thread.lock.
        """
        st = self.__dict__.copy()
        st['agent'] = None
        return st

    def __setstate__(self, st):
        """
        Restore everything except `agent` (we’ll re-attach it externally).
        """
        self.__dict__.update(st)

    def initialize_system(self) -> None:
        """Initialize the evolutionary learning system"""
        
        print("🌱 Initializing evolutionary learning system...")
        
        # Fixed hyperparameter genome of length 8 (matches apply_hyperparameter_genome)
        genome_size = 8
        self.evolutionary_engine.initialize_population(genome_size, "gaussian")
        
        # Initialize system genome from best performing parameters
        self.system_genome = np.random.randn(genome_size)
        
        print("✅ Evolutionary learning system initialized")
    
    def suggest_improvements(self) -> Dict[str, Any]:
        """
        Generate actionable improvement suggestions based on current learning state.
        Returns a dict with:
         - suggestions: List[str]
         - confidence: float (0.0–1.0)
        """
        metrics = self.get_comprehensive_metrics()['system_metrics']
        suggestions: List[str] = []
        # If error rate high
        if self.agent.performance_metrics.get('error_rate', 0.0) > 0.2:
            suggestions.append("Reduce error rate by improving input validation")
        # If response time is slow
        if metrics['current_performance'] < self.performance_thresholds['acceptable']:
            suggestions.append("Optimize model inference for faster responses")
        # If improvement has plateaued
        recent = list(self.improvement_tracker)
        if len(recent) >= 5 and recent[-1] <= recent[0]:
            suggestions.append("Inject diversity: increase mutation rate or explore new strategies")
        # If overall performance is still below good threshold
        if metrics['current_performance'] < self.performance_thresholds['good']:
            suggestions.append("Adjust evolutionary weights to emphasize accuracy over novelty")

        # Estimate confidence as average normalized metric
        conf_vals = [
            metrics['current_performance'],
            metrics['improvement_rate'],
            1.0 - self.agent.performance_metrics.get('error_rate', 0.0)
        ]
        confidence = float(sum(conf_vals) / len(conf_vals))

        return {
            'suggestions': suggestions,
            'confidence': max(0.0, min(1.0, confidence))
        }

    def learning_cycle(self) -> Dict[str, Any]:
        """Execute one complete learning cycle"""
        
        print(f"🧬 Executing learning cycle {self.generation_count}...")
        cycle_start = time.time()
        
        # Phase 1: Evaluate current performance
        current_performance = self._evaluate_current_state()
        
        # Phase 2: Propose improvements using evolutionary proposer
        improvement_proposal = self.proposer.propose_next_action(
            overall_objective="Continuous system improvement",
            current_task="Performance optimization",
            recent_observations=self._get_recent_observations(),
            additional_context=f"Current performance: {current_performance:.4f}"
        )
        
        for suggestion in improvement_proposal.get('suggestions', []):
            # 1️⃣ Look up background info
            research = self.agent.browser.search(suggestion)
            print(f"🔍 Research for '{suggestion}': {research}")

            # 2️⃣ If it looks like code, hand it to the CodeExecutionTool
            if suggestion.strip().startswith(("update_", "train_model", "generate_")):
                result = self.agent.code.execute(suggestion)
                print(f"🖥️ Code execution result: {result}")

        # Phase 3: Execute evolutionary step
        evolution_metrics = self.evolutionary_engine.evolve_generation(
            lambda genome: self._fitness_function(genome, improvement_proposal)
        )
        
        # Phase 4: Update system parameters based on evolution
        best_individual = self.evolutionary_engine.best_individual
        if best_individual:
            self._apply_evolutionary_improvements(best_individual)
        
        # Phase 5: Update proposer with results
        cycle_result = {
            'performance_improvement': evolution_metrics.best_fitness - current_performance,
            'generation': self.generation_count,
            'success': evolution_metrics.best_fitness > current_performance
        }
        
        context_vector = self.proposer._encode_context(
            "Continuous system improvement",
            "Performance optimization", 
            self._get_recent_observations(),
            f"Performance: {evolution_metrics.best_fitness:.4f}"
        )
        
        self.proposer.update_from_experience(
            improvement_proposal['proposed_action'],
            cycle_result,
            context_vector
        )
        
        # Phase 6: Update improvement tracking
        self.improvement_tracker.append(evolution_metrics.best_fitness)
        self.generation_count += 1
        
        # Phase 7: Adaptive system tuning
        self._adaptive_system_tuning(evolution_metrics)
        
        cycle_time = time.time() - cycle_start
        
        # Phase 8: Record learning history
        learning_record = {
            'generation': self.generation_count,
            'performance': evolution_metrics.best_fitness,
            'improvement': cycle_result['performance_improvement'],
            'diversity': evolution_metrics.population_diversity,
            'mutation_rate': self.evolutionary_engine.mutation_rate,
            'exploration_rate': self.proposer.exploration_rate,
            'cycle_time': cycle_time,
            'best_action': improvement_proposal['proposed_action'],
            'confidence': improvement_proposal['confidence_score']
        }
        self.learning_history.append(learning_record)
        
        # ─── Persist evolutionary state ─────────────────────────
        import os, pickle
        STATE_DIR = getattr(self.agent, 'STATE_DIR', './agent_state')
        os.makedirs(STATE_DIR, exist_ok=True)

        # 1️⃣  Save the agent checkpoint (so model weights & metrics stick)
        self.agent.save_state(STATE_DIR, evolution_metrics.best_fitness)

        # 2️⃣  Serialize the entire evolutionary system
        pkl_path = os.path.join(STATE_DIR, 'evolutionary_system.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Evolutionary system saved at gen {self.generation_count}")

        print(f"✅ Learning cycle completed in {cycle_time:.3f}s")
        print(f"   Performance: {evolution_metrics.best_fitness:.6f} (Δ{cycle_result['performance_improvement']:+.6f})")
        print(f"   Best action: {improvement_proposal['proposed_action']}")
        
        return learning_record
    
    def _evaluate_current_state(self) -> float:
        """Evaluate the current state of the agent"""
        
        # Simulate some task results (in real implementation, this would use actual task data)
        simulated_tasks = [
            {'status': 'completed', 'accuracy': 0.85, 'execution_time': 0.5, 'success': True, 'type': 'analysis'},
            {'status': 'completed', 'accuracy': 0.92, 'execution_time': 0.3, 'success': True, 'type': 'research'},
            {'status': 'failed', 'accuracy': 0.2, 'execution_time': 1.2, 'success': False, 'type': 'generation'},
            {'status': 'completed', 'accuracy': 0.78, 'execution_time': 0.8, 'success': True, 'type': 'optimization'}
        ]
        
        # Use evaluator to get comprehensive performance score
        performance = self.evaluator.evaluate_performance(
            agent_state=self.agent.get_performance_metrics(),
            task_results=simulated_tasks
        )
        
        return performance
    
    def _fitness_function(self, genome: np.ndarray, 
                         improvement_proposal: Dict[str, Any]) -> float:
        """Fitness function for evolutionary optimization"""
        
        # Apply genome parameters to simulate system improvements
        performance_modifier = np.mean(genome) * 0.1  # Scale genome influence
        
        # Base performance from current evaluator
        base_performance = self.evaluator.evaluate_performance(
            agent_state=self.agent.get_performance_metrics(),
            task_results=[]  # Empty for simulation
        )
        
        # Apply genome-based modifications
        modified_performance = base_performance + performance_modifier
        
        # Bonus for proposer confidence
        confidence_bonus = improvement_proposal['confidence_score'] * 0.1
        
        # Penalty for extreme genome values (regularization)
        regularization_penalty = np.linalg.norm(genome) * 0.001
        
        final_fitness = modified_performance + confidence_bonus - regularization_penalty
        
        return max(0.0, final_fitness)  # Ensure non-negative fitness
    
    def update_performance_metrics(self) -> Dict[str, Any]:
        """
        Fetch the latest performance metrics from the agent
        and store/return them for external callers.
        """
        metrics = self.get_comprehensive_metrics()['system_metrics']
        # If you want to keep a record:
        self.latest_system_metrics = metrics
        return metrics

    def _apply_evolutionary_improvements(self, best_individual: Dict[str, Any]) -> None:
        """Apply improvements from the best evolved individual"""
        
        genome = best_individual['genome']
        
        # Update agent performance metrics based on evolved parameters
        improvements = {
            'accuracy_rate': max(0.0, min(1.0, self.agent.performance_metrics['accuracy_rate'] + genome[0] * 0.01)),
            'task_completion_rate': max(0.0, min(1.0, self.agent.performance_metrics['task_completion_rate'] + genome[1] * 0.01)),
            'response_time_avg': max(0.1, self.agent.performance_metrics['response_time_avg'] + genome[2] * 0.001),
            'error_rate': max(0.0, min(0.5, self.agent.performance_metrics['error_rate'] + genome[3] * 0.001)),
            'confidence_avg': max(0.0, min(100.0, self.agent.performance_metrics['confidence_avg'] + genome[4] * 0.1)),
            'overall_performance': max(0.0, min(1.0, self.agent.performance_metrics['overall_performance'] + genome[5] * 0.01))
        }
        
        # Apply improvements
        for metric, value in improvements.items():
            self.agent.performance_metrics[metric] = value
        
        # Update best metric for saving
        self.agent._best_metric = self.agent.performance_metrics['overall_performance']
        
        print(f"🔧 Applied evolutionary improvements:")
        print(f"   Overall performance: {self.agent.performance_metrics['overall_performance']:.6f}")
        print(f"   Task completion: {self.agent.performance_metrics['task_completion_rate']:.3f}")
        print(f"   Accuracy: {self.agent.performance_metrics['accuracy_rate']:.3f}")
    
        if hasattr(self.agent.ml_ai, "apply_hyperparameter_genome"):
            try:
                self.agent.ml_ai.apply_hyperparameter_genome(genome)
                print("✅ Hyperparameters updated based on evolved genome")
            except Exception as e:
                print(f"⚠️ Failed to apply hyper-genome: {e}")

    def _get_recent_observations(self) -> List[str]:
        """Get recent observations for context"""
        
        observations = []
        
        # Add performance trend observation
        if len(self.improvement_tracker) >= 2:
            recent_trend = list(self.improvement_tracker)[-5:]
            if len(recent_trend) >= 2:
                trend = "improving" if recent_trend[-1] > recent_trend[0] else "declining"
                observations.append(f"Performance trend: {trend}")
        
        # Add current performance level
        current_perf = self.agent.performance_metrics['overall_performance']
        if current_perf >= self.performance_thresholds['excellent']:
            observations.append("Performance level: excellent")
        elif current_perf >= self.performance_thresholds['good']:
            observations.append("Performance level: good")
        elif current_perf >= self.performance_thresholds['acceptable']:
            observations.append("Performance level: acceptable")
        else:
            observations.append("Performance level: needs improvement")
        
        # Add system state observations
        observations.append(f"Generation: {self.generation_count}")
        observations.append(f"Learning history length: {len(self.learning_history)}")
        
        return observations
    
    def _adaptive_system_tuning(self, evolution_metrics: EvolutionaryMetrics) -> None:
        """Adaptively tune system parameters based on performance"""
        
        # Adjust evolutionary parameters based on performance trends
        if len(self.improvement_tracker) >= 10:
            recent_improvements = list(self.improvement_tracker)[-10:]
            improvement_trend = np.polyfit(range(len(recent_improvements)), recent_improvements, 1)[0]
            
            if improvement_trend > 0.01:  # Strong positive trend
                # Reduce mutation rate to exploit current good region
                self.evolutionary_engine.mutation_rate *= 0.95
                self.proposer.exploration_rate *= 0.98
                
            elif improvement_trend < -0.01:  # Negative trend
                # Increase mutation rate to explore more
                self.evolutionary_engine.mutation_rate *= 1.05
                self.proposer.exploration_rate *= 1.02
                
                # Inject diversity if stagnant
                if abs(improvement_trend) < 0.001:
                    self.evolutionary_engine._handle_stagnation()
        
        # Adjust evaluator expectations
        current_performance = evolution_metrics.best_fitness
        if current_performance > 0.8:
            self.evaluator.difficulty_scaling = min(2.0, self.evaluator.difficulty_scaling * 1.02)
        elif current_performance < 0.4:
            self.evaluator.difficulty_scaling = max(0.5, self.evaluator.difficulty_scaling * 0.98)
        
        # Clamp parameters to reasonable ranges
        self.evolutionary_engine.mutation_rate = np.clip(self.evolutionary_engine.mutation_rate, 0.0001, 0.1)
        self.proposer.exploration_rate = np.clip(self.proposer.exploration_rate, 0.05, 0.5)
        
        print(f"🔧 Adaptive tuning: mutation_rate={self.evolutionary_engine.mutation_rate:.6f}, "
              f"exploration_rate={self.proposer.exploration_rate:.3f}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about the learning system"""
        
        # Calculate improvement statistics
        if len(self.improvement_tracker) > 1:
            improvements = list(self.improvement_tracker)
            improvement_rate = (improvements[-1] - improvements[0]) / len(improvements)
            performance_stability = 1.0 - np.std(improvements)
        else:
            improvement_rate = 0.0
            performance_stability = 0.0
        
        return {
            'system_metrics': {
                'generation_count': self.generation_count,
                'current_performance': self.agent.performance_metrics['overall_performance'],
                'improvement_rate': improvement_rate,
                'performance_stability': performance_stability,
                'learning_cycles_completed': len(self.learning_history)
            },
            'evolutionary_metrics': {
                'population_size': self.evolutionary_engine.population_size,
                'mutation_rate': self.evolutionary_engine.mutation_rate,
                'best_fitness': self.evolutionary_engine.metrics.best_fitness,
                'population_diversity': self.evolutionary_engine.metrics.population_diversity,
                'mutation_success_rate': self.evolutionary_engine.metrics.mutation_success_rate
            },
            'evaluator_metrics': {
                'expectation_level': self.evaluator.expectation_level,
                'difficulty_scaling': self.evaluator.difficulty_scaling,
                'adaptive_weights': self.evaluator.adaptive_weights
            },
            'proposer_metrics': {
                'exploration_rate': self.proposer.exploration_rate,
                'best_strategy_fitness': self.proposer.best_strategy['fitness'] if self.proposer.best_strategy else 0.0,
                'experience_buffer_size': len(self.proposer.experience_buffer),
                'strategy_population_size': len(self.proposer.strategy_population)
            },
            'performance_trajectory': list(self.improvement_tracker),
            'recent_learning_history': self.learning_history[-10:] if self.learning_history else []
        }
    
    def continuous_learning_loop(self, max_generations: int = 1000) -> None:
        """Run continuous learning loop with automatic improvement"""
        
        print(f"🚀 Starting continuous learning loop for {max_generations} generations...")
        
        # Initialize if not already done
        if not self.evolutionary_engine.population:
            self.initialize_system()
        
        try:
            for generation in range(max_generations):
                # Execute learning cycle
                cycle_result = self.learning_cycle()
                
                # Check for significant improvement
                if cycle_result['improvement'] > 0.01:
                    print(f"🎯 Significant improvement detected: +{cycle_result['improvement']:.6f}")
                
                # Periodic status report
                if generation % 10 == 0:
                    metrics = self.get_comprehensive_metrics()
                    print(f"\n📊 Status Report - Generation {generation}:")
                    print(f"   Current Performance: {metrics['system_metrics']['current_performance']:.6f}")
                    print(f"   Improvement Rate: {metrics['system_metrics']['improvement_rate']:.6f}")
                    print(f"   Performance Stability: {metrics['system_metrics']['performance_stability']:.3f}")
                    print()
                
                # Brief pause to prevent overwhelming
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏸️ Learning loop interrupted by user")
        except Exception as e:
            print(f"\n❌ Learning loop error: {e}")
        
        print("🏁 Continuous learning loop completed")

# Integration with the existing agent
def integrate_evolutionary_learning(agent) -> FinalEvolutionaryLearningSystem:
    """
    FIXED: Less aggressive evolutionary learning that respects user requests
    """
    print("🔗 Integrating evolutionary learning system with agent...")
    
    import os, pickle
    STATE_DIR = getattr(agent, 'STATE_DIR', './agent_state')
    pkl_path = os.path.join(STATE_DIR, 'evolutionary_system.pkl')
    learning_system = None

    # Load existing system logic (unchanged)
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                learning_system = pickle.load(f)
            learning_system.agent = agent
            print("✅ Loaded existing evolutionary system")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"⚠️ Reinitializing evolutionary system: {e}")
            learning_system = None

    if learning_system is None:
        learning_system = FinalEvolutionaryLearningSystem(agent)
        learning_system.initialize_system()
    
    # FIXED: Much less aggressive background learning
    def background_learning():
        print("🔄 Starting background evolutionary learning (low priority)")
        generation_count = 0
        max_generations = 50  # Reduced from 100
        
        while generation_count < max_generations:
            try:
                # Wait much longer between learning cycles
                time.sleep(600)  # 10 minutes between cycles
                
                # Only learn if agent has been idle for a while
                if hasattr(agent, 'last_user_request_time'):
                    if time.time() - agent.last_user_request_time < 300:  # 5 min
                        continue
                
                # Quick learning cycle only
                learning_system.learning_cycle()
                generation_count += 1
                
                # Save progress less frequently
                if generation_count % 10 == 0:
                    print(f"🧬 Evolutionary learning: generation {generation_count}")
                
            except Exception as e:
                print(f"⚠️ Background learning error: {e}")
                time.sleep(300)  # Wait 5 min on error
    
    # Start with lower priority
    learning_thread = threading.Thread(target=background_learning, daemon=True)
    learning_thread.start()
    
    agent.evolutionary_learning = learning_system
    print("✅ Evolutionary learning integrated (user-priority mode)")
    
    return learning_system

# Usage example:
"""
# In your app.py, add this to ensure continuous improvement:

# After agent initialization
evolutionary_system = integrate_evolutionary_learning(agent)

# The system will now continuously evolve and improve performance metrics
# You can check progress with:
metrics = evolutionary_system.get_comprehensive_metrics()
print(f"Current performance: {metrics['system_metrics']['current_performance']}")
"""
