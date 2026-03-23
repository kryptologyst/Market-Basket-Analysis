"""
Market Basket Analysis - Evaluation Module

This module provides comprehensive evaluation metrics and business KPIs
for market basket analysis results.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class MarketBasketEvaluator:
    """Evaluates market basket analysis results with business and ML metrics."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary containing evaluation settings.
        """
        self.config = config
        self.metrics = {}
        self.business_kpis = {}
    
    def evaluate(
        self,
        association_rules: pd.DataFrame,
        frequent_itemsets: pd.DataFrame,
        transactions: List[List[str]],
        catalog_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of market basket analysis results.
        
        Args:
            association_rules: DataFrame with association rules.
            frequent_itemsets: DataFrame with frequent itemsets.
            transactions: Original transaction data.
            catalog_df: Optional catalog DataFrame for business metrics.
            
        Returns:
            Dictionary containing all evaluation metrics and KPIs.
        """
        logger.info("Starting comprehensive evaluation")
        
        results = {
            "ml_metrics": self._calculate_ml_metrics(association_rules, frequent_itemsets),
            "business_kpis": self._calculate_business_kpis(
                association_rules, transactions, catalog_df
            ),
            "rule_quality": self._assess_rule_quality(association_rules),
            "coverage_analysis": self._analyze_coverage(association_rules, transactions),
            "novelty_analysis": self._analyze_novelty(association_rules, transactions),
        }
        
        logger.info("Evaluation completed")
        return results
    
    def _calculate_ml_metrics(
        self,
        association_rules: pd.DataFrame,
        frequent_itemsets: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Calculate machine learning metrics.
        
        Args:
            association_rules: DataFrame with association rules.
            frequent_itemsets: DataFrame with frequent itemsets.
            
        Returns:
            Dictionary with ML metrics.
        """
        if len(association_rules) == 0:
            return {"error": "No association rules to evaluate"}
        
        metrics = {
            "total_rules": len(association_rules),
            "total_itemsets": len(frequent_itemsets),
            "avg_support": association_rules["support"].mean(),
            "avg_confidence": association_rules["confidence"].mean(),
            "avg_lift": association_rules["lift"].mean(),
            "avg_conviction": association_rules["conviction"].mean(),
            "max_lift": association_rules["lift"].max(),
            "min_lift": association_rules["lift"].min(),
            "high_confidence_rules": len(
                association_rules[association_rules["confidence"] >= 0.8]
            ),
            "high_lift_rules": len(
                association_rules[association_rules["lift"] >= 2.0]
            ),
            "strong_rules": len(
                association_rules[
                    (association_rules["confidence"] >= 0.7) &
                    (association_rules["lift"] >= 1.5)
                ]
            ),
        }
        
        # Calculate additional metrics
        metrics.update({
            "confidence_std": association_rules["confidence"].std(),
            "lift_std": association_rules["lift"].std(),
            "support_std": association_rules["support"].std(),
            "conviction_std": association_rules["conviction"].std(),
        })
        
        return metrics
    
    def _calculate_business_kpis(
        self,
        association_rules: pd.DataFrame,
        transactions: List[List[str]],
        catalog_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Calculate business KPIs and insights.
        
        Args:
            association_rules: DataFrame with association rules.
            transactions: Original transaction data.
            catalog_df: Optional catalog DataFrame.
            
        Returns:
            Dictionary with business KPIs.
        """
        if len(association_rules) == 0:
            return {"error": "No association rules to evaluate"}
        
        kpis = {
            "cross_sell_potential": self._calculate_cross_sell_potential(association_rules),
            "inventory_optimization": self._calculate_inventory_optimization(
                association_rules, transactions
            ),
            "layout_optimization": self._calculate_layout_optimization(association_rules),
            "promotion_effectiveness": self._calculate_promotion_effectiveness(
                association_rules, catalog_df
            ),
        }
        
        return kpis
    
    def _calculate_cross_sell_potential(self, association_rules: pd.DataFrame) -> Dict[str, Any]:
        """Calculate cross-selling potential metrics.
        
        Args:
            association_rules: DataFrame with association rules.
            
        Returns:
            Dictionary with cross-sell metrics.
        """
        # High-confidence rules for cross-selling
        cross_sell_rules = association_rules[
            (association_rules["confidence"] >= 0.6) &
            (association_rules["lift"] >= 1.2)
        ]
        
        return {
            "total_cross_sell_opportunities": len(cross_sell_rules),
            "avg_cross_sell_confidence": cross_sell_rules["confidence"].mean(),
            "avg_cross_sell_lift": cross_sell_rules["lift"].mean(),
            "top_cross_sell_items": self._get_top_cross_sell_items(cross_sell_rules),
        }
    
    def _calculate_inventory_optimization(
        self,
        association_rules: pd.DataFrame,
        transactions: List[List[str]],
    ) -> Dict[str, Any]:
        """Calculate inventory optimization metrics.
        
        Args:
            association_rules: DataFrame with association rules.
            transactions: Original transaction data.
            
        Returns:
            Dictionary with inventory optimization metrics.
        """
        # Analyze co-occurrence patterns for inventory planning
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        
        co_occurrence_matrix = self._build_co_occurrence_matrix(transactions, all_items)
        
        return {
            "total_unique_items": len(all_items),
            "high_co_occurrence_pairs": self._count_high_co_occurrence_pairs(
                co_occurrence_matrix, threshold=0.3
            ),
            "inventory_clustering_potential": self._calculate_clustering_potential(
                association_rules
            ),
        }
    
    def _calculate_layout_optimization(self, association_rules: pd.DataFrame) -> Dict[str, Any]:
        """Calculate store layout optimization metrics.
        
        Args:
            association_rules: DataFrame with association rules.
            
        Returns:
            Dictionary with layout optimization metrics.
        """
        # Strong rules for layout optimization
        layout_rules = association_rules[
            (association_rules["support"] >= 0.05) &
            (association_rules["confidence"] >= 0.7)
        ]
        
        return {
            "layout_optimization_rules": len(layout_rules),
            "avg_layout_rule_support": layout_rules["support"].mean(),
            "layout_clusters": self._identify_layout_clusters(layout_rules),
        }
    
    def _calculate_promotion_effectiveness(
        self,
        association_rules: pd.DataFrame,
        catalog_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Calculate promotion effectiveness metrics.
        
        Args:
            association_rules: DataFrame with association rules.
            catalog_df: Optional catalog DataFrame.
            
        Returns:
            Dictionary with promotion effectiveness metrics.
        """
        # Rules suitable for promotions (high lift, moderate support)
        promotion_rules = association_rules[
            (association_rules["lift"] >= 1.5) &
            (association_rules["support"] >= 0.02) &
            (association_rules["confidence"] >= 0.5)
        ]
        
        return {
            "promotion_candidate_rules": len(promotion_rules),
            "avg_promotion_lift": promotion_rules["lift"].mean(),
            "promotion_impact_potential": self._calculate_promotion_impact(
                promotion_rules, catalog_df
            ),
        }
    
    def _assess_rule_quality(self, association_rules: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of association rules.
        
        Args:
            association_rules: DataFrame with association rules.
            
        Returns:
            Dictionary with rule quality metrics.
        """
        if len(association_rules) == 0:
            return {"error": "No rules to assess"}
        
        # Rule length distribution
        rule_lengths = association_rules["antecedents"].apply(len) + \
                      association_rules["consequents"].apply(len)
        
        return {
            "avg_rule_length": rule_lengths.mean(),
            "max_rule_length": rule_lengths.max(),
            "min_rule_length": rule_lengths.min(),
            "rules_by_length": rule_lengths.value_counts().to_dict(),
            "quality_score": self._calculate_quality_score(association_rules),
        }
    
    def _analyze_coverage(
        self,
        association_rules: pd.DataFrame,
        transactions: List[List[str]],
    ) -> Dict[str, Any]:
        """Analyze coverage of association rules.
        
        Args:
            association_rules: DataFrame with association rules.
            transactions: Original transaction data.
            
        Returns:
            Dictionary with coverage analysis.
        """
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        
        covered_items = set()
        for _, rule in association_rules.iterrows():
            covered_items.update(rule["antecedents"])
            covered_items.update(rule["consequents"])
        
        return {
            "total_items": len(all_items),
            "covered_items": len(covered_items),
            "coverage_percentage": len(covered_items) / len(all_items) * 100,
            "uncovered_items": list(all_items - covered_items),
        }
    
    def _analyze_novelty(
        self,
        association_rules: pd.DataFrame,
        transactions: List[List[str]],
    ) -> Dict[str, Any]:
        """Analyze novelty of association rules.
        
        Args:
            association_rules: DataFrame with association rules.
            transactions: Original transaction data.
            
        Returns:
            Dictionary with novelty analysis.
        """
        # Calculate how often rule patterns appear in transactions
        rule_frequencies = []
        
        for _, rule in association_rules.iterrows():
            pattern = list(rule["antecedents"]) + list(rule["consequents"])
            frequency = sum(
                1 for transaction in transactions
                if all(item in transaction for item in pattern)
            )
            rule_frequencies.append(frequency)
        
        return {
            "avg_pattern_frequency": np.mean(rule_frequencies),
            "novel_patterns": sum(1 for freq in rule_frequencies if freq <= 2),
            "common_patterns": sum(1 for freq in rule_frequencies if freq >= 10),
        }
    
    def _get_top_cross_sell_items(self, cross_sell_rules: pd.DataFrame) -> List[Tuple[str, float]]:
        """Get top cross-sell items by frequency.
        
        Args:
            cross_sell_rules: DataFrame with cross-sell rules.
            
        Returns:
            List of tuples (item, frequency).
        """
        item_counts = {}
        
        for _, rule in cross_sell_rules.iterrows():
            for item in rule["consequents"]:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        return sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _build_co_occurrence_matrix(
        self,
        transactions: List[List[str]],
        all_items: set,
    ) -> pd.DataFrame:
        """Build co-occurrence matrix for items.
        
        Args:
            transactions: List of transactions.
            all_items: Set of all unique items.
            
        Returns:
            DataFrame with co-occurrence matrix.
        """
        items_list = list(all_items)
        n_items = len(items_list)
        matrix = np.zeros((n_items, n_items))
        
        for transaction in transactions:
            transaction_items = [items_list.index(item) for item in transaction if item in all_items]
            
            for i in transaction_items:
                for j in transaction_items:
                    if i != j:
                        matrix[i][j] += 1
        
        return pd.DataFrame(matrix, index=items_list, columns=items_list)
    
    def _count_high_co_occurrence_pairs(
        self,
        co_occurrence_matrix: pd.DataFrame,
        threshold: float,
    ) -> int:
        """Count pairs with high co-occurrence.
        
        Args:
            co_occurrence_matrix: Co-occurrence matrix.
            threshold: Threshold for high co-occurrence.
            
        Returns:
            Number of high co-occurrence pairs.
        """
        total_transactions = co_occurrence_matrix.sum().sum() / 2  # Divide by 2 for symmetric matrix
        normalized_matrix = co_occurrence_matrix / total_transactions
        
        # Count pairs above threshold (excluding diagonal)
        mask = (normalized_matrix > threshold) & (normalized_matrix.index != normalized_matrix.columns)
        return mask.sum().sum() // 2  # Divide by 2 for symmetric pairs
    
    def _calculate_clustering_potential(self, association_rules: pd.DataFrame) -> float:
        """Calculate inventory clustering potential.
        
        Args:
            association_rules: DataFrame with association rules.
            
        Returns:
            Clustering potential score.
        """
        if len(association_rules) == 0:
            return 0.0
        
        # Calculate average rule strength
        avg_strength = (association_rules["confidence"] * association_rules["lift"]).mean()
        
        # Calculate rule density (rules per unique item)
        all_items = set()
        for _, rule in association_rules.iterrows():
            all_items.update(rule["antecedents"])
            all_items.update(rule["consequents"])
        
        density = len(association_rules) / len(all_items) if len(all_items) > 0 else 0
        
        return avg_strength * density
    
    def _identify_layout_clusters(self, layout_rules: pd.DataFrame) -> List[List[str]]:
        """Identify clusters of items for layout optimization.
        
        Args:
            layout_rules: DataFrame with layout optimization rules.
            
        Returns:
            List of item clusters.
        """
        # Simple clustering based on rule connections
        clusters = []
        processed_items = set()
        
        for _, rule in layout_rules.iterrows():
            rule_items = set(rule["antecedents"]) | set(rule["consequents"])
            
            if not any(item in processed_items for item in rule_items):
                clusters.append(list(rule_items))
                processed_items.update(rule_items)
        
        return clusters
    
    def _calculate_promotion_impact(
        self,
        promotion_rules: pd.DataFrame,
        catalog_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Calculate potential promotion impact.
        
        Args:
            promotion_rules: DataFrame with promotion rules.
            catalog_df: Optional catalog DataFrame.
            
        Returns:
            Dictionary with promotion impact metrics.
        """
        if len(promotion_rules) == 0:
            return {"impact_score": 0.0}
        
        # Calculate impact score based on lift and support
        impact_scores = promotion_rules["lift"] * promotion_rules["support"]
        
        return {
            "avg_impact_score": impact_scores.mean(),
            "max_impact_score": impact_scores.max(),
            "high_impact_rules": len(impact_scores[impact_scores >= impact_scores.quantile(0.8)]),
        }
    
    def _calculate_quality_score(self, association_rules: pd.DataFrame) -> float:
        """Calculate overall quality score for rules.
        
        Args:
            association_rules: DataFrame with association rules.
            
        Returns:
            Quality score between 0 and 1.
        """
        if len(association_rules) == 0:
            return 0.0
        
        # Weighted combination of metrics
        confidence_score = association_rules["confidence"].mean()
        lift_score = min(association_rules["lift"].mean() / 3.0, 1.0)  # Normalize lift
        support_score = association_rules["support"].mean() * 10  # Scale support
        
        quality_score = (confidence_score * 0.4 + lift_score * 0.4 + support_score * 0.2)
        return min(quality_score, 1.0)
    
    def generate_leaderboard(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate a leaderboard of evaluation metrics.
        
        Args:
            evaluation_results: Results from evaluate method.
            
        Returns:
            DataFrame with leaderboard metrics.
        """
        leaderboard_data = []
        
        # ML Metrics
        ml_metrics = evaluation_results.get("ml_metrics", {})
        leaderboard_data.append({
            "Category": "ML Metrics",
            "Metric": "Total Rules",
            "Value": ml_metrics.get("total_rules", 0),
            "Unit": "count",
            "Higher_is_better": True,
        })
        
        leaderboard_data.append({
            "Category": "ML Metrics",
            "Metric": "Average Confidence",
            "Value": ml_metrics.get("avg_confidence", 0),
            "Unit": "ratio",
            "Higher_is_better": True,
        })
        
        leaderboard_data.append({
            "Category": "ML Metrics",
            "Metric": "Average Lift",
            "Value": ml_metrics.get("avg_lift", 0),
            "Unit": "ratio",
            "Higher_is_better": True,
        })
        
        # Business KPIs
        business_kpis = evaluation_results.get("business_kpis", {})
        cross_sell = business_kpis.get("cross_sell_potential", {})
        leaderboard_data.append({
            "Category": "Business KPIs",
            "Metric": "Cross-sell Opportunities",
            "Value": cross_sell.get("total_cross_sell_opportunities", 0),
            "Unit": "count",
            "Higher_is_better": True,
        })
        
        coverage = evaluation_results.get("coverage_analysis", {})
        leaderboard_data.append({
            "Category": "Business KPIs",
            "Metric": "Item Coverage",
            "Value": coverage.get("coverage_percentage", 0),
            "Unit": "percentage",
            "Higher_is_better": True,
        })
        
        return pd.DataFrame(leaderboard_data)
