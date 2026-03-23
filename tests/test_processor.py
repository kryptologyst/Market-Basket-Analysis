"""
Unit tests for Market Basket Analysis - Data Processor

Tests for data processing, validation, and synthetic data generation.
"""

import pytest
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.processor import DataProcessor


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config_dict = {
        "data": {
            "synthetic": {
                "n_transactions": 1000,
                "n_items": 50,
                "n_customers": 200,
                "avg_basket_size": 3.0,
                "max_basket_size": 8,
                "seed": 42
            },
            "real": {
                "transactions_file": "data/raw/transactions.csv",
                "catalog_file": "data/raw/catalog.csv",
                "customers_file": "data/raw/customers.csv"
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
        ["beer", "bread"]
    ]


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_initialization(self, sample_config):
        """Test DataProcessor initialization."""
        processor = DataProcessor(sample_config)
        assert processor.config == sample_config
        assert processor.transaction_encoder is None
    
    def test_set_random_seeds(self, sample_config):
        """Test random seed setting."""
        processor = DataProcessor(sample_config)
        processor.set_random_seeds(123)
        # Test that seeds are set (no exception should be raised)
        assert True
    
    def test_generate_synthetic_data(self, sample_config):
        """Test synthetic data generation."""
        processor = DataProcessor(sample_config)
        
        transactions, catalog_df, customers_df = processor.generate_synthetic_data(
            n_transactions=100,
            n_items=20,
            n_customers=50
        )
        
        # Check transactions
        assert isinstance(transactions, list)
        assert len(transactions) == 100
        assert all(isinstance(t, list) for t in transactions)
        assert all(isinstance(item, str) for t in transactions for item in t)
        
        # Check catalog
        assert isinstance(catalog_df, pd.DataFrame)
        assert len(catalog_df) == 20
        assert "item_id" in catalog_df.columns
        assert "name" in catalog_df.columns
        assert "category" in catalog_df.columns
        assert "price" in catalog_df.columns
        
        # Check customers
        assert isinstance(customers_df, pd.DataFrame)
        assert len(customers_df) == 50
        assert "customer_id" in customers_df.columns
        assert "cohort_month" in customers_df.columns
    
    def test_generate_catalog(self, sample_config):
        """Test catalog generation."""
        processor = DataProcessor(sample_config)
        catalog_df = processor._generate_catalog(10)
        
        assert isinstance(catalog_df, pd.DataFrame)
        assert len(catalog_df) == 10
        assert all(col in catalog_df.columns for col in ["item_id", "name", "category", "price", "cost"])
        assert all(catalog_df["price"] > catalog_df["cost"])
    
    def test_generate_customers(self, sample_config):
        """Test customer generation."""
        processor = DataProcessor(sample_config)
        customers_df = processor._generate_customers(10)
        
        assert isinstance(customers_df, pd.DataFrame)
        assert len(customers_df) == 10
        assert all(col in customers_df.columns for col in ["customer_id", "cohort_month", "lifetime_value", "frequency"])
    
    def test_convert_transactions_df_to_list(self, sample_config):
        """Test DataFrame to list conversion."""
        processor = DataProcessor(sample_config)
        
        # Create sample DataFrame
        df = pd.DataFrame({
            "transaction_id": ["trans_1", "trans_1", "trans_2", "trans_2", "trans_3"],
            "item_name": ["milk", "bread", "beer", "bread", "eggs"]
        })
        
        transactions = processor._convert_transactions_df_to_list(df)
        
        assert isinstance(transactions, list)
        assert len(transactions) == 3
        assert ["milk", "bread"] in transactions
        assert ["beer", "bread"] in transactions
        assert ["eggs"] in transactions
    
    def test_convert_transactions_list_to_df(self, sample_config, sample_transactions):
        """Test list to DataFrame conversion."""
        processor = DataProcessor(sample_config)
        
        df = processor._convert_transactions_list_to_df(sample_transactions)
        
        assert isinstance(df, pd.DataFrame)
        assert "transaction_id" in df.columns
        assert "item_name" in df.columns
        assert len(df) == sum(len(t) for t in sample_transactions)
    
    def test_save_data(self, sample_config, sample_transactions, tmp_path):
        """Test data saving functionality."""
        processor = DataProcessor(sample_config)
        
        catalog_df = pd.DataFrame({
            "item_id": ["item_1", "item_2"],
            "name": ["milk", "bread"],
            "category": ["Dairy", "Bakery"],
            "price": [3.99, 2.49],
            "cost": [2.50, 1.25]
        })
        
        customers_df = pd.DataFrame({
            "customer_id": ["customer_1", "customer_2"],
            "cohort_month": ["2023-01", "2023-02"],
            "lifetime_value": [450.50, 320.75],
            "frequency": [12, 8]
        })
        
        processor.save_data(sample_transactions, catalog_df, customers_df, str(tmp_path))
        
        # Check that files were created
        assert (tmp_path / "transactions.csv").exists()
        assert (tmp_path / "catalog.csv").exists()
        assert (tmp_path / "customers.csv").exists()
        
        # Check file contents
        transactions_df = pd.read_csv(tmp_path / "transactions.csv")
        assert len(transactions_df) > 0
        assert "transaction_id" in transactions_df.columns
        assert "item_name" in transactions_df.columns


class TestDataValidation:
    """Test cases for data validation."""
    
    def test_valid_transactions(self, sample_transactions):
        """Test validation of valid transaction data."""
        from src.utils.helpers import validate_transactions
        
        assert validate_transactions(sample_transactions) is True
    
    def test_invalid_transactions_empty_list(self):
        """Test validation of empty transaction list."""
        from src.utils.helpers import validate_transactions
        
        assert validate_transactions([]) is False
    
    def test_invalid_transactions_wrong_type(self):
        """Test validation of wrong data type."""
        from src.utils.helpers import validate_transactions
        
        assert validate_transactions("not a list") is False
    
    def test_invalid_transactions_non_list_items(self):
        """Test validation of transactions with non-list items."""
        from src.utils.helpers import validate_transactions
        
        invalid_transactions = [
            ["milk", "bread"],
            "not a list",
            ["eggs", "butter"]
        ]
        
        assert validate_transactions(invalid_transactions) is False
    
    def test_clean_transactions(self, sample_transactions):
        """Test transaction cleaning."""
        from src.utils.helpers import clean_transactions
        
        # Add some empty items and duplicates
        dirty_transactions = [
            ["milk", "", "bread", "milk", "eggs"],
            ["beer", "bread", "  ", "butter"],
            ["milk", "bread", "bread"]
        ]
        
        cleaned = clean_transactions(dirty_transactions)
        
        assert len(cleaned) == 3
        assert all(len(set(t)) == len(t) for t in cleaned)  # No duplicates
        assert all(item.strip() for t in cleaned for item in t)  # No empty items
    
    def test_calculate_transaction_statistics(self, sample_transactions):
        """Test transaction statistics calculation."""
        from src.utils.helpers import calculate_transaction_statistics
        
        stats = calculate_transaction_statistics(sample_transactions)
        
        assert isinstance(stats, dict)
        assert "total_transactions" in stats
        assert "total_items" in stats
        assert "avg_basket_size" in stats
        assert "most_frequent_items" in stats
        assert "most_co_occurring_pairs" in stats
        
        assert stats["total_transactions"] == len(sample_transactions)
        assert stats["total_items"] > 0
        assert stats["avg_basket_size"] > 0


class TestIntegration:
    """Integration tests for data processing pipeline."""
    
    def test_end_to_end_synthetic_data(self, sample_config, tmp_path):
        """Test complete synthetic data generation and saving."""
        processor = DataProcessor(sample_config)
        
        # Generate data
        transactions, catalog_df, customers_df = processor.generate_synthetic_data(
            n_transactions=50,
            n_items=20,
            n_customers=30
        )
        
        # Save data
        processor.save_data(transactions, catalog_df, customers_df, str(tmp_path))
        
        # Load data back
        loaded_transactions, loaded_catalog, loaded_customers = processor.load_real_data(str(tmp_path))
        
        # Verify data integrity
        assert len(loaded_transactions) == len(transactions)
        assert len(loaded_catalog) == len(catalog_df)
        assert len(loaded_customers) == len(customers_df)
    
    def test_data_consistency(self, sample_config):
        """Test data consistency across different operations."""
        processor = DataProcessor(sample_config)
        
        # Generate data multiple times with same seed
        processor.set_random_seeds(42)
        transactions1, catalog1, customers1 = processor.generate_synthetic_data(n_transactions=100)
        
        processor.set_random_seeds(42)
        transactions2, catalog2, customers2 = processor.generate_synthetic_data(n_transactions=100)
        
        # Should be identical with same seed
        assert transactions1 == transactions2
        assert catalog1.equals(catalog2)
        assert customers1.equals(customers2)
