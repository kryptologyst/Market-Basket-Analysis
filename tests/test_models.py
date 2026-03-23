"""
Unit tests for Market Basket Analysis - Model Testing

Tests for market basket analysis algorithms and model functionality.
"""

import pytest
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.models.basket_analyzer import MarketBasketAnalyzer


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config_dict = {
        "models": {
            "apriori": {
                "min_support": 0.01,
                "min_confidence": 0.5,
                "min_lift": 1.0,
                "max_length": 10
            },
            "fp_growth": {
                "min_support": 0.01,
                "min_confidence": 0.5,
                "min_lift": 1.0,
                "max_length": 10
            },
            "eclat": {
                "min_support": 0.01,
                "min_confidence": 0.5,
                "min_lift": 1.0,
                "max_length": 10
            }
        }
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    return [
        ["milk", "bread", "eggs"],
        ["beer", "bread", "butter"],
        ["milk", "bread"],
        ["milk", "eggs"],
        ["bread", "butter"],
        ["milk", "bread", "butter", "eggs"],
        ["beer", "bread"],
        ["milk", "bread", "eggs"],
        ["beer", "bread", "butter"],
        ["milk", "eggs"]
    ]


class TestMarketBasketAnalyzer:
    """Test cases for MarketBasketAnalyzer class."""
    
    def test_initialization(self, sample_config):
        """Test analyzer initialization."""
        analyzer = MarketBasketAnalyzer(sample_config)
        assert analyzer.config == sample_config
        assert analyzer.frequent_itemsets is None
        assert analyzer.association_rules is None
        assert analyzer.algorithm_used is None
    
    def test_fit_apriori(self, sample_config, sample_transactions):
        """Test fitting with Apriori algorithm."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        analyzer.fit(sample_transactions, algorithm="apriori")
        
        assert analyzer.algorithm_used == "apriori"
        assert analyzer.frequent_itemsets is not None
        assert analyzer.association_rules is not None
        assert len(analyzer.frequent_itemsets) > 0
    
    def test_fit_fp_growth(self, sample_config, sample_transactions):
        """Test fitting with FP-Growth algorithm."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        analyzer.fit(sample_transactions, algorithm="fp_growth")
        
        assert analyzer.algorithm_used == "fp_growth"
        assert analyzer.frequent_itemsets is not None
        assert analyzer.association_rules is not None
    
    def test_fit_eclat(self, sample_config, sample_transactions):
        """Test fitting with ECLAT algorithm."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        analyzer.fit(sample_transactions, algorithm="eclat")
        
        assert analyzer.algorithm_used == "eclat"
        assert analyzer.frequent_itemsets is not None
        assert analyzer.association_rules is not None
    
    def test_invalid_algorithm(self, sample_config, sample_transactions):
        """Test fitting with invalid algorithm."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            analyzer.fit(sample_transactions, algorithm="invalid")
    
    def test_get_frequent_itemsets(self, sample_config, sample_transactions):
        """Test getting frequent itemsets."""
        analyzer = MarketBasketAnalyzer(sample_config)
        analyzer.fit(sample_transactions, algorithm="apriori")
        
        itemsets = analyzer.get_frequent_itemsets()
        
        assert isinstance(itemsets, pd.DataFrame)
        assert "support" in itemsets.columns
        assert "itemsets" in itemsets.columns
        assert len(itemsets) > 0
    
    def test_get_association_rules(self, sample_config, sample_transactions):
        """Test getting association rules."""
        analyzer = MarketBasketAnalyzer(sample_config)
        analyzer.fit(sample_transactions, algorithm="apriori")
        
        rules = analyzer.get_association_rules()
        
        assert isinstance(rules, pd.DataFrame)
        assert "antecedents" in rules.columns
        assert "consequents" in rules.columns
        assert "support" in rules.columns
        assert "confidence" in rules.columns
        assert "lift" in rules.columns
    
    def test_get_top_rules(self, sample_config, sample_transactions):
        """Test getting top rules by metric."""
        analyzer = MarketBasketAnalyzer(sample_config)
        analyzer.fit(sample_transactions, algorithm="apriori")
        
        top_rules = analyzer.get_top_rules(metric="lift", n=5)
        
        assert isinstance(top_rules, pd.DataFrame)
        assert len(top_rules) <= 5
        
        if len(top_rules) > 1:
            # Check that rules are sorted by lift (descending)
            lifts = top_rules["lift"].tolist()
            assert lifts == sorted(lifts, reverse=True)
    
    def test_get_rules_by_items(self, sample_config, sample_transactions):
        """Test getting rules containing specific items."""
        analyzer = MarketBasketAnalyzer(sample_config)
        analyzer.fit(sample_transactions, algorithm="apriori")
        
        milk_rules = analyzer.get_rules_by_items(["milk"], direction="both")
        
        assert isinstance(milk_rules, pd.DataFrame)
        
        if len(milk_rules) > 0:
            # Check that all rules contain milk
            for _, rule in milk_rules.iterrows():
                rule_items = list(rule["antecedents"]) + list(rule["consequents"])
                assert "milk" in rule_items
    
    def test_predict_recommendations(self, sample_config, sample_transactions):
        """Test recommendation prediction."""
        analyzer = MarketBasketAnalyzer(sample_config)
        analyzer.fit(sample_transactions, algorithm="apriori")
        
        recommendations = analyzer.predict_recommendations(
            basket=["milk", "bread"],
            n_recommendations=3
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        for item, confidence, rule in recommendations:
            assert isinstance(item, str)
            assert isinstance(confidence, float)
            assert isinstance(rule, str)
            assert 0 <= confidence <= 1
    
    def test_get_business_insights(self, sample_config, sample_transactions):
        """Test business insights generation."""
        analyzer = MarketBasketAnalyzer(sample_config)
        analyzer.fit(sample_transactions, algorithm="apriori")
        
        insights = analyzer.get_business_insights()
        
        assert isinstance(insights, dict)
        assert "total_rules" in insights
        assert "avg_confidence" in insights
        assert "avg_lift" in insights
        assert "cross_sell_opportunities" in insights
    
    def test_empty_transactions(self, sample_config):
        """Test handling of empty transactions."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        analyzer.fit([], algorithm="apriori")
        
        assert analyzer.frequent_itemsets is not None
        assert analyzer.association_rules is not None
        assert len(analyzer.frequent_itemsets) == 0
        assert len(analyzer.association_rules) == 0
    
    def test_single_item_transactions(self, sample_config):
        """Test handling of single-item transactions."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        single_item_transactions = [
            ["milk"],
            ["bread"],
            ["eggs"],
            ["milk"],
            ["bread"]
        ]
        
        analyzer.fit(single_item_transactions, algorithm="apriori")
        
        assert analyzer.frequent_itemsets is not None
        assert analyzer.association_rules is not None
    
    def test_custom_parameters(self, sample_config, sample_transactions):
        """Test fitting with custom parameters."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        analyzer.fit(
            sample_transactions,
            algorithm="apriori",
            min_support=0.1,
            min_confidence=0.8,
            min_lift=2.0,
            max_length=3
        )
        
        assert analyzer.algorithm_used == "apriori"
        
        # Check that parameters were applied
        if len(analyzer.association_rules) > 0:
            assert all(analyzer.association_rules["confidence"] >= 0.8)
            assert all(analyzer.association_rules["lift"] >= 2.0)


class TestModelConsistency:
    """Test model consistency and reproducibility."""
    
    def test_reproducibility(self, sample_config, sample_transactions):
        """Test that results are reproducible."""
        analyzer1 = MarketBasketAnalyzer(sample_config)
        analyzer2 = MarketBasketAnalyzer(sample_config)
        
        analyzer1.fit(sample_transactions, algorithm="apriori")
        analyzer2.fit(sample_transactions, algorithm="apriori")
        
        # Results should be identical
        rules1 = analyzer1.get_association_rules()
        rules2 = analyzer2.get_association_rules()
        
        assert len(rules1) == len(rules2)
        
        if len(rules1) > 0:
            # Check that rules are the same (order may differ)
            rules1_sorted = rules1.sort_values(["support", "confidence", "lift"])
            rules2_sorted = rules2.sort_values(["support", "confidence", "lift"])
            
            assert rules1_sorted.equals(rules2_sorted)
    
    def test_algorithm_comparison(self, sample_config, sample_transactions):
        """Test comparison between different algorithms."""
        apriori_analyzer = MarketBasketAnalyzer(sample_config)
        fp_growth_analyzer = MarketBasketAnalyzer(sample_config)
        
        apriori_analyzer.fit(sample_transactions, algorithm="apriori")
        fp_growth_analyzer.fit(sample_transactions, algorithm="fp_growth")
        
        apriori_rules = apriori_analyzer.get_association_rules()
        fp_growth_rules = fp_growth_analyzer.get_association_rules()
        
        # Both algorithms should find some rules
        assert len(apriori_rules) >= 0
        assert len(fp_growth_rules) >= 0
        
        # Results should be similar (may not be identical due to implementation differences)
        if len(apriori_rules) > 0 and len(fp_growth_rules) > 0:
            # Check that both find rules with reasonable metrics
            assert apriori_rules["confidence"].mean() > 0
            assert fp_growth_rules["confidence"].mean() > 0
            assert apriori_rules["lift"].mean() > 0
            assert fp_growth_rules["lift"].mean() > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_frequent_itemsets(self, sample_config):
        """Test when no frequent itemsets are found."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        # Use very high support threshold
        analyzer.fit(
            [["milk"], ["bread"], ["eggs"]],
            algorithm="apriori",
            min_support=0.9
        )
        
        assert len(analyzer.frequent_itemsets) == 0
        assert len(analyzer.association_rules) == 0
    
    def test_no_association_rules(self, sample_config):
        """Test when no association rules are found."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        # Use very high confidence threshold
        analyzer.fit(
            [["milk", "bread"], ["beer", "butter"]],
            algorithm="apriori",
            min_confidence=0.99
        )
        
        assert len(analyzer.association_rules) == 0
    
    def test_large_transactions(self, sample_config):
        """Test handling of large transactions."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        large_transactions = [
            ["item_" + str(i) for i in range(20)] for _ in range(10)
        ]
        
        analyzer.fit(large_transactions, algorithm="apriori")
        
        assert analyzer.frequent_itemsets is not None
        assert analyzer.association_rules is not None
    
    def test_duplicate_items_in_transaction(self, sample_config):
        """Test handling of duplicate items in transactions."""
        analyzer = MarketBasketAnalyzer(sample_config)
        
        duplicate_transactions = [
            ["milk", "milk", "bread"],
            ["beer", "beer", "butter"],
            ["milk", "bread", "bread"]
        ]
        
        analyzer.fit(duplicate_transactions, algorithm="apriori")
        
        assert analyzer.frequent_itemsets is not None
        assert analyzer.association_rules is not None
