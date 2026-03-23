"""
Market Basket Analysis - Data Processing Module

This module handles data loading, preprocessing, and synthetic data generation
for market basket analysis.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing for market basket analysis."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize the data processor with configuration.
        
        Args:
            config: Configuration dictionary containing data settings.
        """
        self.config = config
        self.set_random_seeds()
    
    def set_random_seeds(self, seed: Optional[int] = None) -> None:
        """Set random seeds for reproducibility.
        
        Args:
            seed: Random seed. If None, uses config seed.
        """
        if seed is None:
            seed = self.config.data.synthetic.seed
        
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seeds set to {seed}")
    
    def generate_synthetic_data(
        self,
        n_transactions: Optional[int] = None,
        n_items: Optional[int] = None,
        n_customers: Optional[int] = None,
        avg_basket_size: Optional[float] = None,
        max_basket_size: Optional[int] = None,
    ) -> Tuple[List[List[str]], pd.DataFrame, pd.DataFrame]:
        """Generate synthetic transaction data for testing and demonstration.
        
        Args:
            n_transactions: Number of transactions to generate.
            n_items: Number of unique items in the catalog.
            n_customers: Number of unique customers.
            avg_basket_size: Average number of items per basket.
            max_basket_size: Maximum number of items per basket.
            
        Returns:
            Tuple of (transactions, catalog_df, customers_df).
        """
        # Use config defaults if not provided
        n_transactions = n_transactions or self.config.data.synthetic.n_transactions
        n_items = n_items or self.config.data.synthetic.n_items
        n_customers = n_customers or self.config.data.synthetic.n_customers
        avg_basket_size = avg_basket_size or self.config.data.synthetic.avg_basket_size
        max_basket_size = max_basket_size or self.config.data.synthetic.max_basket_size
        
        logger.info(f"Generating {n_transactions} transactions with {n_items} items")
        
        # Generate item catalog
        catalog_df = self._generate_catalog(n_items)
        
        # Generate customer base
        customers_df = self._generate_customers(n_customers)
        
        # Generate transactions with realistic patterns
        transactions = self._generate_transactions(
            catalog_df, customers_df, n_transactions, avg_basket_size, max_basket_size
        )
        
        logger.info(f"Generated {len(transactions)} transactions")
        return transactions, catalog_df, customers_df
    
    def _generate_catalog(self, n_items: int) -> pd.DataFrame:
        """Generate a synthetic product catalog.
        
        Args:
            n_items: Number of items to generate.
            
        Returns:
            DataFrame with item information.
        """
        # Define item categories and their probabilities
        categories = {
            "Dairy": 0.15,
            "Bakery": 0.12,
            "Meat": 0.10,
            "Produce": 0.18,
            "Pantry": 0.20,
            "Beverages": 0.12,
            "Snacks": 0.08,
            "Frozen": 0.05,
        }
        
        items = []
        for i in range(n_items):
            category = np.random.choice(
                list(categories.keys()), p=list(categories.values())
            )
            
            # Generate item name based on category
            item_name = f"{category.lower()}_item_{i+1:03d}"
            
            # Generate realistic price based on category
            base_prices = {
                "Dairy": (2.0, 8.0),
                "Bakery": (1.5, 6.0),
                "Meat": (4.0, 15.0),
                "Produce": (1.0, 5.0),
                "Pantry": (1.0, 12.0),
                "Beverages": (1.5, 8.0),
                "Snacks": (1.0, 6.0),
                "Frozen": (2.0, 10.0),
            }
            
            min_price, max_price = base_prices[category]
            price = round(np.random.uniform(min_price, max_price), 2)
            
            items.append({
                "item_id": f"item_{i+1:03d}",
                "name": item_name,
                "category": category,
                "price": price,
                "cost": round(price * np.random.uniform(0.4, 0.7), 2),
            })
        
        return pd.DataFrame(items)
    
    def _generate_customers(self, n_customers: int) -> pd.DataFrame:
        """Generate a synthetic customer base.
        
        Args:
            n_customers: Number of customers to generate.
            
        Returns:
            DataFrame with customer information.
        """
        customers = []
        for i in range(n_customers):
            customers.append({
                "customer_id": f"customer_{i+1:04d}",
                "cohort_month": np.random.choice([
                    "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
                    "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12"
                ]),
                "lifetime_value": np.random.exponential(500),
                "frequency": np.random.poisson(8),
            })
        
        return pd.DataFrame(customers)
    
    def _generate_transactions(
        self,
        catalog_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        n_transactions: int,
        avg_basket_size: float,
        max_basket_size: int,
    ) -> List[List[str]]:
        """Generate realistic transaction data with association patterns.
        
        Args:
            catalog_df: Product catalog DataFrame.
            customers_df: Customer DataFrame.
            n_transactions: Number of transactions to generate.
            avg_basket_size: Average basket size.
            max_basket_size: Maximum basket size.
            
        Returns:
            List of transactions (each transaction is a list of item names).
        """
        transactions = []
        item_names = catalog_df["name"].tolist()
        
        # Define some realistic association patterns
        associations = {
            ("dairy_item_001", "bakery_item_002"): 0.7,  # milk + bread
            ("dairy_item_001", "bakery_item_003"): 0.6,  # milk + cereal
            ("meat_item_001", "pantry_item_001"): 0.8,  # chicken + rice
            ("produce_item_001", "produce_item_002"): 0.5,  # apples + bananas
            ("beverages_item_001", "snacks_item_001"): 0.6,  # soda + chips
        }
        
        for _ in range(n_transactions):
            # Generate basket size using negative binomial distribution
            basket_size = min(
                max(1, int(np.random.negative_binomial(avg_basket_size, 0.5))),
                max_basket_size
            )
            
            basket = []
            
            # Start with a random item
            first_item = np.random.choice(item_names)
            basket.append(first_item)
            
            # Add items based on associations and random selection
            for _ in range(basket_size - 1):
                # Check for association patterns
                associated_items = []
                for (item1, item2), prob in associations.items():
                    if item1 in basket and np.random.random() < prob:
                        associated_items.append(item2)
                    elif item2 in basket and np.random.random() < prob:
                        associated_items.append(item1)
                
                if associated_items:
                    next_item = np.random.choice(associated_items)
                else:
                    # Random selection with category preference
                    next_item = np.random.choice(item_names)
                
                if next_item not in basket:
                    basket.append(next_item)
            
            transactions.append(basket)
        
        return transactions
    
    def load_real_data(self, data_dir: str) -> Tuple[List[List[str]], pd.DataFrame, pd.DataFrame]:
        """Load real transaction data from files.
        
        Args:
            data_dir: Directory containing data files.
            
        Returns:
            Tuple of (transactions, catalog_df, customers_df).
        """
        data_path = Path(data_dir)
        
        # Load transactions
        transactions_file = data_path / self.config.data.real.transactions_file
        if transactions_file.exists():
            transactions_df = pd.read_csv(transactions_file)
            transactions = self._convert_transactions_df_to_list(transactions_df)
        else:
            logger.warning(f"Transactions file not found: {transactions_file}")
            transactions = []
        
        # Load catalog
        catalog_file = data_path / self.config.data.real.catalog_file
        if catalog_file.exists():
            catalog_df = pd.read_csv(catalog_file)
        else:
            logger.warning(f"Catalog file not found: {catalog_file}")
            catalog_df = pd.DataFrame()
        
        # Load customers
        customers_file = data_path / self.config.data.real.customers_file
        if customers_file.exists():
            customers_df = pd.read_csv(customers_file)
        else:
            logger.warning(f"Customers file not found: {customers_file}")
            customers_df = pd.DataFrame()
        
        return transactions, catalog_df, customers_df
    
    def _convert_transactions_df_to_list(self, transactions_df: pd.DataFrame) -> List[List[str]]:
        """Convert transaction DataFrame to list format.
        
        Args:
            transactions_df: DataFrame with transaction data.
            
        Returns:
            List of transactions.
        """
        # Group by transaction ID and collect items
        transactions = []
        for _, group in transactions_df.groupby("transaction_id"):
            items = group["item_name"].tolist()
            transactions.append(items)
        
        return transactions
    
    def save_data(
        self,
        transactions: List[List[str]],
        catalog_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """Save processed data to files.
        
        Args:
            transactions: List of transactions.
            catalog_df: Catalog DataFrame.
            customers_df: Customers DataFrame.
            output_dir: Output directory path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save transactions
        transactions_df = self._convert_transactions_list_to_df(transactions)
        transactions_df.to_csv(output_path / "transactions.csv", index=False)
        
        # Save catalog
        catalog_df.to_csv(output_path / "catalog.csv", index=False)
        
        # Save customers
        customers_df.to_csv(output_path / "customers.csv", index=False)
        
        logger.info(f"Data saved to {output_path}")
    
    def _convert_transactions_list_to_df(self, transactions: List[List[str]]) -> pd.DataFrame:
        """Convert transaction list to DataFrame format.
        
        Args:
            transactions: List of transactions.
            
        Returns:
            DataFrame with transaction data.
        """
        data = []
        for i, transaction in enumerate(transactions):
            for item in transaction:
                data.append({
                    "transaction_id": f"trans_{i+1:06d}",
                    "item_name": item,
                })
        
        return pd.DataFrame(data)
