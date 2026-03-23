# Market Basket Analysis

A comprehensive market basket analysis toolkit for identifying product associations and generating business insights. This project implements multiple association rule mining algorithms (Apriori, FP-Growth, ECLAT) with advanced evaluation metrics and interactive visualizations.

## Important Disclaimer

**This is a research and educational tool for market basket analysis. Results should not be used for automated business decisions without human review and validation. Always verify insights with domain experts before implementing any recommendations.**

## Features

- **Multiple Algorithms**: Apriori, FP-Growth, and ECLAT implementations
- **Comprehensive Evaluation**: ML metrics, business KPIs, coverage analysis, and novelty assessment
- **Interactive Visualizations**: Plotly-based charts and dashboards
- **Streamlit Demo**: User-friendly web interface for analysis
- **Synthetic Data Generation**: Realistic transaction data for testing
- **Business Insights**: Cross-selling, inventory optimization, layout planning, and promotion effectiveness
- **Production Ready**: Type hints, logging, configuration management, and comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Market-Basket-Analysis.git
cd Market-Basket-Analysis

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

#### Command Line Interface

```bash
# Generate synthetic data
python scripts/run_analysis.py --generate-data --data-dir data/processed

# Run analysis with Apriori algorithm
python scripts/run_analysis.py --run-analysis --algorithm apriori --data-dir data/processed --output-dir assets/results

# Run with custom parameters
python scripts/run_analysis.py --run-analysis --algorithm fp_growth --min-support 0.02 --min-confidence 0.6 --min-lift 1.5
```

#### Python API

```python
from omegaconf import OmegaConf
from src.data.processor import DataProcessor
from src.models.basket_analyzer import MarketBasketAnalyzer
from src.eval.evaluator import MarketBasketEvaluator

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Initialize components
processor = DataProcessor(config)
analyzer = MarketBasketAnalyzer(config)
evaluator = MarketBasketEvaluator(config)

# Generate synthetic data
transactions, catalog_df, customers_df = processor.generate_synthetic_data()

# Run analysis
analyzer.fit(transactions, algorithm="apriori")
frequent_itemsets = analyzer.get_frequent_itemsets()
association_rules = analyzer.get_association_rules()

# Evaluate results
evaluation_results = evaluator.evaluate(association_rules, frequent_itemsets, transactions)

# Get business insights
insights = analyzer.get_business_insights()
print(f"Found {len(association_rules)} association rules")
print(f"Cross-sell opportunities: {insights['cross_sell_opportunities']}")
```

#### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
market-basket-analysis/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   └── processor.py          # Data loading and generation
│   ├── models/                   # Model implementations
│   │   └── basket_analyzer.py    # Market basket analysis algorithms
│   ├── eval/                     # Evaluation metrics
│   │   └── evaluator.py          # Comprehensive evaluation
│   ├── viz/                      # Visualizations
│   │   └── visualizer.py         # Plotly-based charts
│   └── utils/                    # Utility functions
│       └── helpers.py            # Helper functions
├── configs/                      # Configuration files
│   └── config.yaml               # Main configuration
├── scripts/                      # Command-line scripts
│   └── run_analysis.py           # Main analysis script
├── demo/                         # Interactive demo
│   └── app.py                    # Streamlit application
├── tests/                        # Unit tests
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── assets/                       # Analysis results
├── notebooks/                    # Jupyter notebooks
├── pyproject.toml                # Project configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files for easy customization. Key settings include:

- **Data settings**: Synthetic data parameters, file paths
- **Model settings**: Algorithm parameters (support, confidence, lift thresholds)
- **Evaluation settings**: Metrics and business KPIs to calculate
- **Visualization settings**: Plot styles, colors, figure sizes
- **Logging settings**: Log levels and output files

Example configuration:

```yaml
data:
  synthetic:
    n_transactions: 10000
    n_items: 100
    avg_basket_size: 4.5

models:
  apriori:
    min_support: 0.01
    min_confidence: 0.5
    min_lift: 1.0
```

## Algorithms

### Apriori Algorithm
- Classic association rule mining algorithm
- Uses breadth-first search with candidate generation
- Good for small to medium datasets
- Configurable support and confidence thresholds

### FP-Growth Algorithm
- Frequent Pattern Growth algorithm
- Uses FP-tree data structure
- More efficient than Apriori for large datasets
- Reduces memory usage and computation time

### ECLAT Algorithm
- Equivalence Class Clustering and bottom-up Lattice Traversal
- Uses vertical data format
- Efficient for dense datasets
- Good for finding closed itemsets

## Evaluation Metrics

### Machine Learning Metrics
- **Support**: Frequency of itemset occurrence
- **Confidence**: Conditional probability of consequent given antecedent
- **Lift**: Ratio of observed support to expected support
- **Conviction**: Measure of rule strength
- **Coverage**: Percentage of items covered by rules
- **Novelty**: Measure of rule uniqueness

### Business KPIs
- **Cross-sell Potential**: Opportunities for selling additional products
- **Inventory Optimization**: Insights for inventory management
- **Layout Optimization**: Recommendations for store layout
- **Promotion Effectiveness**: Effectiveness of promotional campaigns

## Data Format

### Transaction Data
The system expects transaction data in the following format:

```csv
transaction_id,item_name
trans_000001,milk
trans_000001,bread
trans_000001,eggs
trans_000002,beer
trans_000002,bread
```

### Catalog Data (Optional)
```csv
item_id,name,category,price,cost
item_001,milk,Dairy,3.99,2.50
item_002,bread,Bakery,2.49,1.25
```

### Customer Data (Optional)
```csv
customer_id,cohort_month,lifetime_value,frequency
customer_0001,2023-01,450.50,12
customer_0002,2023-02,320.75,8
```

## Visualization

The project provides comprehensive visualizations:

- **Association Rules Plot**: Top rules by various metrics
- **3D Scatter Plot**: Support vs Confidence vs Lift
- **Itemset Support Distribution**: Histogram of support values
- **Rule Length Distribution**: Distribution of rule complexities
- **Business KPIs Dashboard**: Key business metrics
- **Coverage Analysis**: Item coverage visualization
- **Comprehensive Dashboard**: All visualizations in one view

## Business Applications

### Cross-selling
- Identify products frequently bought together
- Create targeted cross-selling campaigns
- Optimize product recommendations

### Inventory Management
- Understand product co-occurrence patterns
- Optimize inventory levels
- Reduce stockouts and overstock

### Store Layout
- Place related products near each other
- Optimize customer flow
- Increase basket size and customer satisfaction

### Promotional Campaigns
- Identify effective product combinations for promotions
- Measure promotion impact
- Optimize promotional strategies

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/ scripts/ demo/
ruff check src/ tests/ scripts/ demo/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_processor.py
```

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for code quality

## Limitations and Considerations

1. **Data Quality**: Results depend heavily on data quality and completeness
2. **Temporal Effects**: Current implementation doesn't account for temporal patterns
3. **Seasonality**: Seasonal effects are not explicitly modeled
4. **External Factors**: External factors (promotions, events) are not considered
5. **Scalability**: Performance may vary with dataset size and complexity
6. **Interpretability**: Complex rules may be difficult to interpret and implement

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{market_basket_analysis,
  title={Market Basket Analysis Toolkit},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Market-Basket-Analysis}
}
```

## Support

For questions, issues, or contributions, please:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Contact the maintainers

## Changelog

### Version 1.0.0
- Initial release
- Apriori, FP-Growth, and ECLAT algorithms
- Comprehensive evaluation metrics
- Interactive Streamlit demo
- Synthetic data generation
- Business KPI calculations
- Production-ready code structure
# Market-Basket-Analysis
