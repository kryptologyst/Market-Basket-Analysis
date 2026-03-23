"""
Market Basket Analysis Package

A comprehensive toolkit for market basket analysis with multiple algorithms,
evaluation metrics, and interactive visualizations.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "assistant@example.com"

from .data.processor import DataProcessor
from .models.basket_analyzer import MarketBasketAnalyzer
from .eval.evaluator import MarketBasketEvaluator
from .viz.visualizer import MarketBasketVisualizer

__all__ = [
    "DataProcessor",
    "MarketBasketAnalyzer", 
    "MarketBasketEvaluator",
    "MarketBasketVisualizer",
]
