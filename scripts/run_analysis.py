"""
Market Basket Analysis - Main Script

Command-line interface for running market basket analysis with various algorithms
and generating comprehensive reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.processor import DataProcessor
from src.models.basket_analyzer import MarketBasketAnalyzer
from src.eval.evaluator import MarketBasketEvaluator
from src.viz.visualizer import MarketBasketVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(config: DictConfig) -> None:
    """Setup logging configuration."""
    log_config = config.get("logging", {})
    
    # Create logs directory
    log_file = log_config.get("file", "logs/market_basket.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    file_handler.setFormatter(formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    return OmegaConf.load(config_path)


def generate_synthetic_data(config: DictConfig, output_dir: str) -> None:
    """Generate synthetic data for analysis."""
    logger.info("Generating synthetic data")
    
    processor = DataProcessor(config)
    
    # Generate data
    transactions, catalog_df, customers_df = processor.generate_synthetic_data()
    
    # Save data
    processor.save_data(transactions, catalog_df, customers_df, output_dir)
    
    logger.info(f"Synthetic data saved to {output_dir}")


def run_analysis(
    config: DictConfig,
    data_dir: str,
    algorithm: str,
    output_dir: str,
    min_support: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_lift: Optional[float] = None,
    max_length: Optional[int] = None,
) -> None:
    """Run market basket analysis."""
    logger.info(f"Starting market basket analysis with {algorithm} algorithm")
    
    # Initialize components
    processor = DataProcessor(config)
    analyzer = MarketBasketAnalyzer(config)
    evaluator = MarketBasketEvaluator(config)
    visualizer = MarketBasketVisualizer(config)
    
    # Load data
    logger.info("Loading data")
    transactions, catalog_df, customers_df = processor.load_real_data(data_dir)
    
    if not transactions:
        logger.error("No transaction data found")
        return
    
    # Run analysis
    logger.info("Running analysis")
    analyzer.fit(
        transactions=transactions,
        algorithm=algorithm,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_length=max_length,
    )
    
    # Get results
    frequent_itemsets = analyzer.get_frequent_itemsets()
    association_rules = analyzer.get_association_rules()
    
    logger.info(f"Found {len(frequent_itemsets)} frequent itemsets and {len(association_rules)} association rules")
    
    # Run evaluation
    logger.info("Running evaluation")
    evaluation_results = evaluator.evaluate(
        association_rules=association_rules,
        frequent_itemsets=frequent_itemsets,
        transactions=transactions,
        catalog_df=catalog_df,
    )
    
    # Generate visualizations
    logger.info("Generating visualizations")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    frequent_itemsets.to_csv(output_path / "frequent_itemsets.csv", index=False)
    association_rules.to_csv(output_path / "association_rules.csv", index=False)
    
    # Generate leaderboard
    leaderboard = evaluator.generate_leaderboard(evaluation_results)
    leaderboard.to_csv(output_path / "evaluation_leaderboard.csv", index=False)
    
    # Generate visualizations
    if len(association_rules) > 0:
        # Top rules plot
        fig = visualizer.plot_association_rules(association_rules, top_n=20)
        fig.write_html(output_path / "top_association_rules.html")
        
        # 3D scatter plot
        fig = visualizer.plot_support_confidence_lift(association_rules)
        fig.write_html(output_path / "support_confidence_lift.html")
        
        # Business KPIs dashboard
        fig = visualizer.plot_business_kpis(evaluation_results)
        fig.write_html(output_path / "business_kpis.html")
        
        # Comprehensive dashboard
        fig = visualizer.create_comprehensive_dashboard(
            association_rules, frequent_itemsets, evaluation_results
        )
        fig.write_html(output_path / "comprehensive_dashboard.html")
    
    # Generate report
    generate_report(evaluation_results, output_path / "analysis_report.txt")
    
    logger.info(f"Analysis completed. Results saved to {output_dir}")


def generate_report(evaluation_results: Dict[str, Any], report_path: Path) -> None:
    """Generate a comprehensive analysis report."""
    logger.info("Generating analysis report")
    
    with open(report_path, "w") as f:
        f.write("Market Basket Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # ML Metrics
        ml_metrics = evaluation_results.get("ml_metrics", {})
        f.write("Machine Learning Metrics:\n")
        f.write("-" * 25 + "\n")
        for metric, value in ml_metrics.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        f.write("\n")
        
        # Business KPIs
        business_kpis = evaluation_results.get("business_kpis", {})
        f.write("Business KPIs:\n")
        f.write("-" * 15 + "\n")
        for kpi, details in business_kpis.items():
            f.write(f"{kpi}:\n")
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {details}\n")
        f.write("\n")
        
        # Coverage Analysis
        coverage = evaluation_results.get("coverage_analysis", {})
        f.write("Coverage Analysis:\n")
        f.write("-" * 18 + "\n")
        for metric, value in coverage.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        f.write("\n")
        
        # Novelty Analysis
        novelty = evaluation_results.get("novelty_analysis", {})
        f.write("Novelty Analysis:\n")
        f.write("-" * 17 + "\n")
        for metric, value in novelty.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        f.write("\n")
        
        # Recommendations
        f.write("Recommendations:\n")
        f.write("-" * 16 + "\n")
        f.write("1. Focus on high-confidence rules for cross-selling campaigns\n")
        f.write("2. Use high-lift rules for store layout optimization\n")
        f.write("3. Consider support values for inventory planning\n")
        f.write("4. Monitor rule quality metrics for model validation\n")
        f.write("5. Implement A/B testing for rule-based recommendations\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Market Basket Analysis Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing transaction data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["apriori", "fp_growth", "eclat"],
        default="apriori",
        help="Association rule mining algorithm"
    )
    parser.add_argument(
        "--min-support",
        type=float,
        help="Minimum support threshold"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        help="Minimum lift threshold"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum itemset length"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic data"
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run market basket analysis"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    # Generate synthetic data if requested
    if args.generate_data:
        generate_synthetic_data(config, args.data_dir)
    
    # Run analysis if requested
    if args.run_analysis:
        run_analysis(
            config=config,
            data_dir=args.data_dir,
            algorithm=args.algorithm,
            output_dir=args.output_dir,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift,
            max_length=args.max_length,
        )
    
    # If no specific action requested, show help
    if not args.generate_data and not args.run_analysis:
        parser.print_help()


if __name__ == "__main__":
    main()
