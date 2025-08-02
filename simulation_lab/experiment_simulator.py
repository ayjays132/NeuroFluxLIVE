# SelfResearch/simulation_lab/experiment_simulator.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os
from typing import List, Dict, Any, Union

from utils.colors import Colors
from utils.tensor_ops import tensor_to_ndarray

# --- Configure Logging for structured and colored output ---
# Custom formatter to add colors
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

# Set up logging to console
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set default level to INFO
if not log.handlers: # Prevent adding multiple handlers if run multiple times
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    log.addHandler(console_handler)

class ExperimentSimulator:
    """
    A class for simulating virtual experiments using PyTorch with CUDA acceleration.
    This can be extended for various scientific simulations (physics, biology, etc.).
    Designed for production readiness with robust error handling, detailed logging,
    and clear visualization capabilities.
    """
    def __init__(self, device: str = 'cpu'):
        """
        Initializes the ExperimentSimulator with a specified device.
        Args:
            device (str): The device to use for computations ("cuda" or "cpu").
        Raises:
            ValueError: If the specified device is not supported.
            RuntimeError: If 'cuda' is specified but not available.
        """
        log.info(f"{Colors.BLUE}Initializing ExperimentSimulator with device preference: '{device}'{Colors.RESET}")
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device(device)
                log.info(f"{Colors.GREEN}{Colors.BOLD}ExperimentSimulator initialized on GPU: {self.device}{Colors.RESET}")
            else:
                log.warning(f"{Colors.YELLOW}CUDA is not available. Falling back to CPU.{Colors.RESET}")
                self.device = torch.device('cpu')
                log.info(f"{Colors.BLUE}ExperimentSimulator initialized on CPU (CUDA unavailable).{Colors.RESET}")
        elif device == 'cpu':
            self.device = torch.device('cpu')
            log.info(f"{Colors.BLUE}ExperimentSimulator initialized on CPU.{Colors.RESET}")
        else:
            raise ValueError(f"{Colors.RED}Unsupported device: '{device}'. Please choose 'cpu' or 'cuda'.{Colors.RESET}")

        # Ensure output directory for plots exists
        self.output_dir = "simulation_results"
        os.makedirs(self.output_dir, exist_ok=True)
        log.info(f"{Colors.CYAN}Simulation outputs will be saved in: {self.output_dir}{Colors.RESET}")


    def run_physics_simulation(self, initial_position: float, initial_velocity: float, time_steps: int, dt: float) -> torch.Tensor:
        """
        Simulates a simple 1D physics experiment (e.g., projectile motion under gravity).
        
        Args:
            initial_position (float): Starting position.
            initial_velocity (float): Starting velocity.
            time_steps (int): Number of simulation steps (must be positive).
            dt (float): Time step size (must be positive).
            
        Returns:
            torch.Tensor: Tensor containing positions at each time step.
            
        Raises:
            ValueError: If time_steps or dt are non-positive.
        """
        log.info(f"{Colors.BLUE}Starting 1D Physics Simulation...{Colors.RESET}")
        # --- Input Validation ---
        if time_steps <= 0:
            raise ValueError(f"{Colors.RED}Time steps must be a positive integer. Got: {time_steps}{Colors.RESET}")
        if dt <= 0:
            raise ValueError(f"{Colors.RED}Time step size (dt) must be positive. Got: {dt}{Colors.RESET}")

        g = torch.tensor(9.81, device=self.device, dtype=torch.float64) # Acceleration due to gravity, use float64 for precision
        
        positions = torch.zeros(time_steps, device=self.device, dtype=torch.float64)
        velocities = torch.zeros(time_steps, device=self.device, dtype=torch.float64)
        
        positions[0] = torch.tensor(initial_position, device=self.device, dtype=torch.float64)
        velocities[0] = torch.tensor(initial_velocity, device=self.device, dtype=torch.float64)
        
        for i in range(1, time_steps):
            velocities[i] = velocities[i-1] - g * dt
            positions[i] = positions[i-1] + velocities[i] * dt
            
        log.info(f"{Colors.GREEN}1D Physics Simulation completed.{Colors.RESET}")
        return positions

    def run_biological_simulation(self, initial_population_a: float, initial_population_b: float, growth_rate_a: float, growth_rate_b: float, interaction_rate: float, time_steps: int, dt: float) -> torch.Tensor:
        """
        Simulates a simple Lotka-Volterra predator-prey model.
        
        Args:
            initial_population_a (float): Initial population of species A (prey, must be non-negative).
            initial_population_b (float): Initial population of species B (predator, must be non-negative).
            growth_rate_a (float): Growth rate of species A.
            growth_rate_b (float): Death rate of species B.
            interaction_rate (float): Interaction rate between species A and B.
            time_steps (int): Number of simulation steps (must be positive).
            dt (float): Time step size (must be positive).
            
        Returns:
            torch.Tensor: Tensor containing populations of A and B at each time step.
            
        Raises:
            ValueError: If initial populations, time_steps or dt are invalid.
        """
        log.info(f"{Colors.BLUE}Starting Biological Simulation (Lotka-Volterra)...{Colors.RESET}")
        # --- Input Validation ---
        if initial_population_a < 0 or initial_population_b < 0:
            raise ValueError(f"{Colors.RED}Initial populations must be non-negative.{Colors.RESET}")
        if time_steps <= 0:
            raise ValueError(f"{Colors.RED}Time steps must be a positive integer. Got: {time_steps}{Colors.RESET}")
        if dt <= 0:
            raise ValueError(f"{Colors.RED}Time step size (dt) must be positive. Got: {dt}{Colors.RESET}")

        populations = torch.zeros(time_steps, 2, device=self.device, dtype=torch.float64)
        populations[0, 0] = torch.tensor(initial_population_a, device=self.device, dtype=torch.float64)
        populations[0, 1] = torch.tensor(initial_population_b, device=self.device, dtype=torch.float64)
        
        for i in range(1, time_steps):
            pop_a = populations[i-1, 0]
            pop_b = populations[i-1, 1]

            # --- Numerical Stability Check (pre-calculation) ---
            # Prevent potential NaN/Inf from division by zero or log of zero if model was more complex
            # For Lotka-Volterra, populations can theoretically go to zero, but practically 1e-9 is a safe floor.
            if pop_a < 1e-9: pop_a = torch.tensor(1e-9, device=self.device, dtype=torch.float64)
            if pop_b < 1e-9: pop_b = torch.tensor(1e-9, device=self.device, dtype=torch.float64)

            # Lotka-Volterra equations
            d_pop_a = growth_rate_a * pop_a - interaction_rate * pop_a * pop_b
            d_pop_b = interaction_rate * pop_a * pop_b - growth_rate_b * pop_b
            
            populations[i, 0] = pop_a + d_pop_a * dt
            populations[i, 1] = pop_b + d_pop_b * dt
            
            # Ensure populations don't go below zero
            populations[i, 0] = torch.max(populations[i, 0], torch.tensor(0.0, device=self.device, dtype=torch.float64))
            populations[i, 1] = torch.max(populations[i, 1], torch.tensor(0.0, device=self.device, dtype=torch.float64))

            # --- Post-calculation NaN/Inf check ---
            if torch.isnan(populations[i]).any() or torch.isinf(populations[i]).any():
                log.error(f"{Colors.RED}Numerical instability detected at step {i}. Population became NaN/Inf.{Colors.RESET}")
                return populations[:i] # Return up to the point of instability
            
        log.info(f"{Colors.GREEN}Biological Simulation completed.{Colors.RESET}")
        return populations

    def generate_synthetic_data(self, num_samples: int, num_features: int, noise_level: float = 0.1) -> torch.Tensor:
        """
        Generates synthetic dataset with configurable parameters.
        
        Args:
            num_samples (int): Number of data samples (must be positive).
            num_features (int): Number of features per sample (must be positive).
            noise_level (float): Standard deviation of added noise (must be non-negative).
            
        Returns:
            torch.Tensor: Generated synthetic data.
            
        Raises:
            ValueError: If num_samples, num_features, or noise_level are invalid.
        """
        log.info(f"{Colors.BLUE}Generating Synthetic Data (Samples: {num_samples}, Features: {num_features}, Noise: {noise_level})...{Colors.RESET}")
        # --- Input Validation ---
        if num_samples <= 0:
            raise ValueError(f"{Colors.RED}Number of samples must be positive. Got: {num_samples}{Colors.RESET}")
        if num_features <= 0:
            raise ValueError(f"{Colors.RED}Number of features must be positive. Got: {num_features}{Colors.RESET}")
        if noise_level < 0:
            raise ValueError(f"{Colors.RED}Noise level cannot be negative. Got: {noise_level}{Colors.RESET}")

        # Generate random data
        data = torch.randn(num_samples, num_features, device=self.device, dtype=torch.float64)
        # Add some noise
        if noise_level > 0:
            noise = torch.randn(num_samples, num_features, device=self.device, dtype=torch.float64) * noise_level
            synthetic_data = data + noise
        else:
            synthetic_data = data
            
        log.info(f"{Colors.GREEN}Synthetic data generated with shape: {synthetic_data.shape}{Colors.RESET}")
        return synthetic_data

    def visualize_data(self, data: torch.Tensor, title: str = "Simulation Data", labels: List[str] = None):
        """
        Visualizes 1D or 2D data using matplotlib and saves to a file.
        
        Args:
            data (torch.Tensor): Data to visualize.
            title (str): Title for the plot.
            labels (List[str]): Optional list of labels for different data series (e.g., for biological populations)
                                 or axis labels for 1D/2D data.
        """
        log.info(f"{Colors.BLUE}Generating visualization: '{title}'...{Colors.RESET}")
        try:
            data_np = tensor_to_ndarray(data)  # Move to CPU for plotting
            
            plt.figure(figsize=(12, 7)) # Larger figure for better detail
            
            if data_np.ndim == 1:
                plt.plot(data_np, color='cyan', linewidth=2, alpha=0.8)
                plt.xlabel(labels[0] if labels and len(labels) > 0 else "Time Step or Sample Index", fontsize=12, color='white')
                plt.ylabel("Value" if not labels or len(labels) <= 1 else labels[1], fontsize=12, color='white') # Use labels[1] if available for 1D data's Y
            elif data_np.ndim == 2:
                if data_np.shape[1] == 2:
                    if labels and len(labels) == 2:
                        # Plotting two series (e.g., predator/prey)
                        plt.plot(data_np[:, 0], label=labels[0], color='lime', linewidth=2, alpha=0.8)
                        plt.plot(data_np[:, 1], label=labels[1], color='magenta', linewidth=2, alpha=0.8, linestyle='--')
                        plt.legend(fontsize=10, loc='upper right', frameon=False, labelcolor='white')
                        plt.xlabel("Time Step", fontsize=12, color='white')
                        plt.ylabel("Population", fontsize=12, color='white')
                    else:
                        # Scatter plot for 2 features
                        plt.scatter(data_np[:, 0], data_np[:, 1], c='blue', s=10, alpha=0.6, edgecolors='none')
                        plt.xlabel(labels[0] if labels and len(labels) > 0 else "Feature 1", fontsize=12, color='white')
                        plt.ylabel(labels[1] if labels and len(labels) > 1 else "Feature 2", fontsize=12, color='white')
                else:
                    log.warning(f"{Colors.YELLOW}Visualization currently supports only 1D data or 2D data with 2 features for scatter/line plot. Skipping visualization.{Colors.RESET}")
                    return
            else:
                log.warning(f"{Colors.YELLOW}Visualization currently supports only 1D or 2D data. Data has {data_np.ndim} dimensions. Skipping visualization.{Colors.RESET}")
                return
            
            plt.title(f"{title} (Device: {self.device})", fontsize=14, fontweight='bold', color='white')
            plt.grid(True, linestyle=':', alpha=0.6, color='gray')
            
            # Set plot background and tick colors
            ax = plt.gca()
            ax.set_facecolor('#1a1a2e') # Dark background for futuristic feel
            plt.gcf().set_facecolor('#1a1a2e') # Figure background
            ax.tick_params(colors='white') # White ticks

            # Remove problematic lines that attempt to set label color without text argument
            # plt.ylabel(color='white') # Removed
            # plt.xlabel(color='white') # Removed
            
            plt.tight_layout() # Adjust plot to ensure everything fits

            # Save path
            sanitized_title = title.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').lower()
            file_path = os.path.join(self.output_dir, f"{sanitized_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e') # High DPI, tight bbox, dark background
            plt.close()
            log.info(f"{Colors.GREEN}Visualization saved to: {file_path}{Colors.RESET}")
        except Exception as e:
            log.error(f"{Colors.RED}Failed to generate visualization for '{title}': {e}{Colors.RESET}")

    def simulate(self, simulation_type: str, **kwargs) -> torch.Tensor:
        """
        A unified method to run various types of simulations based on the provided type and parameters.

        Args:
            simulation_type (str): The type of simulation to run.
                                   Supported types: 'physics', 'biological', 'synthetic_data'.
            **kwargs: Keyword arguments specific to the chosen simulation type.

        Returns:
            torch.Tensor: The results of the simulation.

        Raises:
            ValueError: If an unsupported simulation_type is provided or
                        if required parameters for a simulation type are missing.
        """
        log.info(f"{Colors.MAGENTA}Initiating unified simulation: '{simulation_type}'{Colors.RESET}")
        if simulation_type == "physics":
            required_params = ["initial_position", "initial_velocity", "time_steps", "dt"]
            if not all(param in kwargs for param in required_params):
                raise ValueError(
                    f"{Colors.RED}Missing parameters for physics simulation. "
                    f"Required: {', '.join(required_params)}{Colors.RESET}"
                )
            return self.run_physics_simulation(
                initial_position=kwargs["initial_position"],
                initial_velocity=kwargs["initial_velocity"],
                time_steps=kwargs["time_steps"],
                dt=kwargs["dt"]
            )
        elif simulation_type == "biological":
            required_params = [
                "initial_population_a", "initial_population_b", "growth_rate_a",
                "growth_rate_b", "interaction_rate", "time_steps", "dt"
            ]
            if not all(param in kwargs for param in required_params):
                raise ValueError(
                    f"{Colors.RED}Missing parameters for biological simulation. "
                    f"Required: {', '.join(required_params)}{Colors.RESET}"
                )
            return self.run_biological_simulation(
                initial_population_a=kwargs["initial_population_a"],
                initial_population_b=kwargs["initial_population_b"],
                growth_rate_a=kwargs["growth_rate_a"],
                growth_rate_b=kwargs["growth_rate_b"],
                interaction_rate=kwargs["interaction_rate"],
                time_steps=kwargs["time_steps"],
                dt=kwargs["dt"]
            )
        elif simulation_type == "synthetic_data":
            required_params = ["num_samples", "num_features"]
            if not all(param in kwargs for param in required_params):
                raise ValueError(
                    f"{Colors.RED}Missing parameters for synthetic data generation. "
                    f"Required: {', '.join(required_params)}{Colors.RESET}"
                )
            return self.generate_synthetic_data(
                num_samples=kwargs["num_samples"],
                num_features=kwargs["num_features"],
                noise_level=kwargs.get("noise_level", 0.1) # Optional parameter with default
            )
        else:
            raise ValueError(
                f"{Colors.RED}Unsupported simulation type: '{simulation_type}'. "
                f"Choose from 'physics', 'biological', 'synthetic_data'.{Colors.RESET}"
            )


if __name__ == "__main__":
    # --- Experiment Orchestration ---
    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║           🌌 Quantum Leap Simulation Lab - Initializing...             ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}\n")

    # Determine optimal device
    selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        simulator = ExperimentSimulator(device=selected_device)

        # Example 1: Run a Physics Simulation using the unified 'simulate' method
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- [UNIFIED SIMULATION: PHYSICS] Quantum Particle Trajectory ---{Colors.RESET}")
        physics_params = {
            "initial_position": 0.0,
            "initial_velocity": 25.0,
            "time_steps": 500,
            "dt": 0.05
        }
        positions = simulator.simulate(simulation_type="physics", **physics_params)
        # Display a readable snippet of the simulation output
        log.info(f"{Colors.GREEN}Physics Simulation Results (first 5 positions): {positions[:5].tolist()}{Colors.RESET}")
        log.info(f"{Colors.GREEN}Physics Simulation Results (last 5 positions): {positions[-5:].tolist()}{Colors.RESET}")
        simulator.visualize_data(positions, "Quantum Particle Trajectory Simulation", labels=["Position (m)"])

        # Example 2: Run a Biological Simulation using the unified 'simulate' method
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- [UNIFIED SIMULATION: BIOLOGY] Interstellar Ecosystem Dynamics ---{Colors.RESET}")
        biological_params = {
            "initial_population_a": 500.0,
            "initial_population_b": 50.0,
            "growth_rate_a": 0.2,
            "growth_rate_b": 0.08,
            "interaction_rate": 0.0005,
            "time_steps": 400,
            "dt": 0.25
        }
        populations = simulator.simulate(simulation_type="biological", **biological_params)
        if populations is not None:
            # Display a readable snippet of the simulation output
            log.info(f"{Colors.GREEN}Biological Simulation Results (first 5 population pairs): {populations[:5].tolist()}{Colors.RESET}")
            log.info(f"{Colors.GREEN}Biological Simulation Results (last 5 population pairs): {populations[-5:].tolist()}{Colors.RESET}")
            simulator.visualize_data(populations, "Interstellar Ecosystem (Lotka-Volterra)", labels=["Exo-Prey Population", "Exo-Predator Population"])

        # Example 3: Generate Synthetic Data using the unified 'simulate' method
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- [UNIFIED SIMULATION: SYNTHETIC DATA] Hyperspectral Data Cube Synthesis ---{Colors.RESET}")
        synthetic_data_params = {
            "num_samples": 2000,
            "num_features": 2,
            "noise_level": 0.05
        }
        synthetic_data = simulator.simulate(simulation_type="synthetic_data", **synthetic_data_params)
        log.info(f"{Colors.GREEN}Generated synthetic data shape: {synthetic_data.shape}{Colors.RESET}")
        # Display a readable snippet of the synthetic data
        log.info(f"{Colors.GREEN}Synthetic Data (first 3 samples): {synthetic_data[:3].tolist()}{Colors.RESET}")
        simulator.visualize_data(synthetic_data, "Hyperspectral Data Scatter Plot (Synthetic)", labels=["Spectral Band 1 (Intensity)", "Spectral Band 2 (Intensity)"])

        # Example 4: Generate 1D synthetic data for time-series analysis using the unified 'simulate' method
        log.info(f"\n{Colors.CYAN}{Colors.BOLD}--- [UNIFIED SIMULATION: SYNTHETIC DATA] Sensor Reading Time-Series ---{Colors.RESET}")
        synthetic_data_1d_params = {
            "num_samples": 1000,
            "num_features": 1,
            "noise_level": 0.1
        }
        synthetic_data_1d = simulator.simulate(simulation_type="synthetic_data", **synthetic_data_1d_params)
        # Display a readable snippet of the 1D synthetic data
        log.info(f"{Colors.GREEN}Generated 1D synthetic data shape: {synthetic_data_1d.shape}{Colors.RESET}")
        log.info(f"{Colors.GREEN}1D Sensor Data (first 5 readings): {synthetic_data_1d.squeeze()[:5].tolist()}{Colors.RESET}")
        simulator.visualize_data(synthetic_data_1d.squeeze(), "Synthetic 1D Time-Series Data Plot", labels=["Time Step", "Sensor Reading (Units)"])


    except ValueError as ve:
        log.critical(f"{Colors.RED}Configuration Error: {ve}{Colors.RESET}")
    except RuntimeError as re:
        log.critical(f"{Colors.RED}Runtime Environment Error: {re}. Ensure PyTorch is correctly installed for your device.{Colors.RESET}")
    except Exception as e:
        log.critical(f"{Colors.RED}An unexpected error occurred: {e}{Colors.RESET}")

    log.info(f"\n{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╔═════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}║               Simulation Lab Operations Concluded.                      ║{Colors.RESET}")
    log.info(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}╚═════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")