"""Utility Helper Module

Provides general-purpose helper functions and utility classes.
"""

import json
import pickle
import logging
import random
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import hashlib


def generate_timestamped_filename(base_name: str, extension: str = ".json", include_hash: bool = False, hash_data: Any = None) -> str:
    """Generate a unique filename with timestamp

    Args:
        base_name: Base filename
        extension: File extension
        include_hash: Whether to include hash value
        hash_data: Data used to generate hash

    Returns:
        Timestamped filename
    """
    timestamp = create_timestamp()
    filename = f"{base_name}_{timestamp}"
    
    if include_hash and hash_data is not None:
        hash_value = generate_hash(hash_data)[:8]  # Use first 8 chars of hash
        filename += f"_{hash_value}"
    
    return filename + extension


def save_object_with_timestamp(obj: Any, base_name: str, directory: Union[str, Path] = ".", extension: str = None, include_hash: bool = False) -> Path:
    """Save object to a timestamped file

    Args:
        obj: Object to save
        base_name: Base filename
        directory: Save directory
        extension: File extension (defaults to .json if not specified)
        include_hash: Whether to include object hash in filename

    Returns:
        Saved file path
    """
    if extension is None:
        extension = ".json"
    
    filename = generate_timestamped_filename(base_name, extension, include_hash, obj)
    filepath = Path(directory) / filename
    
    save_object(obj, filepath)
    return filepath


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("beer_game")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def save_object(obj: Any, filepath: Union[str, Path]) -> None:
    """Save object to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == '.json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_object(filepath: Union[str, Path]) -> Any:
    """Load object from file"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate basic statistics"""
    if not data:
        return {}
    
    data_array = np.array(data)
    return {
        'mean': float(np.mean(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'median': float(np.median(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75)),
        'cv': float(np.std(data_array) / np.mean(data_array)) if np.mean(data_array) != 0 else 0
    }


def moving_average(data: List[float], window: int) -> List[float]:
    """Calculate moving average"""
    if len(data) < window:
        return data.copy()
    
    result = []
    for i in range(len(data)):
        if i < window - 1:
            result.append(data[i])
        else:
            avg = sum(data[i-window+1:i+1]) / window
            result.append(avg)
    
    return result


def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
    """Exponential smoothing"""
    if not data:
        return []
    
    result = [data[0]]
    for i in range(1, len(data)):
        smoothed = alpha * data[i] + (1 - alpha) * result[-1]
        result.append(smoothed)
    
    return result


def generate_hash(data: Any) -> str:
    """Generate data hash value"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(data_str.encode()).hexdigest()


def format_number(num: float, precision: int = 2) -> str:
    """Format number for display"""
    if abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration completeness"""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")
    return True


def create_timestamp() -> str:
    """Create timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division, avoiding division by zero"""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value within the specified range"""
    return max(min_val, min(value, max_val))


def interpolate_missing_values(data: List[Optional[float]]) -> List[float]:
    """Interpolate missing values"""
    result = []
    last_valid = None

    for i, value in enumerate(data):
        if value is not None:
            result.append(value)
            last_valid = value
        else:
            # Find next valid value
            next_valid = None
            for j in range(i + 1, len(data)):
                if data[j] is not None:
                    next_valid = data[j]
                    break

            # Interpolation calculation
            if last_valid is not None and next_valid is not None:
                # Linear interpolation
                steps = j - i + 1
                step_size = (next_valid - last_valid) / steps
                interpolated = last_valid + step_size * (i - data.index(last_valid))
                result.append(interpolated)
            elif last_valid is not None:
                # Use last valid value
                result.append(last_valid)
            elif next_valid is not None:
                # Use next valid value
                result.append(next_valid)
            else:
                # Use default value
                result.append(0.0)

    return result


class PerformanceTimer:
    """Performance timer"""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.logger = logging.getLogger("beer_game")

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"{self.name} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"{self.name} completed, duration: {duration:.2f}s")

    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ConfigValidator:
    """Configuration validator"""

    @staticmethod
    def validate_agent_config(config: Dict[str, Any]) -> bool:
        """Validate agent configuration"""
        required_keys = ['agent_type', 'initial_inventory']
        validate_config(config, required_keys)

        if config['initial_inventory'] < 0:
            raise ValueError("Initial inventory cannot be negative")

        return True

    @staticmethod
    def validate_simulation_config(config: Dict[str, Any]) -> bool:
        """Validate simulation configuration"""
        required_keys = ['total_weeks', 'lead_time']
        validate_config(config, required_keys)

        if config['total_weeks'] <= 0:
            raise ValueError("Total weeks must be greater than 0")

        if config['lead_time'] < 0:
            raise ValueError("Lead time cannot be negative")

        return True

    @staticmethod
    def validate_demand_config(config: Dict[str, Any]) -> bool:
        """Validate demand configuration"""
        required_keys = ['pattern_type', 'base_demand']
        validate_config(config, required_keys)

        if config['base_demand'] < 0:
            raise ValueError("Base demand cannot be negative")

        return True


class DataExporter:
    """Data exporter"""

    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
        """Export data to CSV file"""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
        except ImportError:
            # Use csv module if pandas is not available
            import csv

            if not data:
                return

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    @staticmethod
    def export_to_excel(data: Dict[str, List[Dict[str, Any]]], filepath: Union[str, Path]) -> None:
        """Export data to Excel file (multiple sheets)"""
        try:
            import pandas as pd

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for sheet_name, sheet_data in data.items():
                    if sheet_data:
                        df = pd.DataFrame(sheet_data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
        except ImportError:
            raise ImportError("pandas and openpyxl required to export Excel files")


# Common constants
DEFAULT_COLORS = {
    'retailer': '#FF6B6B',
    'wholesaler': '#4ECDC4',
    'distributor': '#45B7D1',
    'manufacturer': '#96CEB4',
    'demand': '#FECA57',
    'cost': '#FF9FF3'
}

DEFAULT_AGENT_NAMES = {
    'retailer': 'Retailer',
    'wholesaler': 'Wholesaler',
    'distributor': 'Distributor',
    'manufacturer': 'Manufacturer'
}