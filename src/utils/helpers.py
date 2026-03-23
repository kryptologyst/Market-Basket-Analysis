"""
Market Basket Analysis - Utility Functions

Common utility functions for data processing, validation, and analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_transactions(transactions: List[List[str]]) -> bool:
    """Validate transaction data format.
    
    Args:
        transactions: List of transactions to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(transactions, list):
        logger.error("Transactions must be a list")
        return False
    
    if len(transactions) == 0:
        logger.error("Transactions list is empty")
        return False
    
    for i, transaction in enumerate(transactions):
        if not isinstance(transaction, list):
            logger.error(f"Transaction {i} is not a list")
            return False
        
        if len(transaction) == 0:
            logger.warning(f"Transaction {i} is empty")
        
        for j, item in enumerate(transaction):
            if not isinstance(item, str):
                logger.error(f"Item {j} in transaction {i} is not a string")
                return False
            
            if not item.strip():
                logger.warning(f"Empty item in transaction {i}")
    
    logger.info(f"Validated {len(transactions)} transactions")
    return True


def clean_transactions(transactions: List[List[str]]) -> List[List[str]]:
    """Clean transaction data by removing empty items and duplicates.
    
    Args:
        transactions: List of transactions to clean.
        
    Returns:
        Cleaned list of transactions.
    """
    cleaned_transactions = []
    
    for transaction in transactions:
        # Remove empty strings and strip whitespace
        cleaned_transaction = [item.strip() for item in transaction if item.strip()]
        
        # Remove duplicates while preserving order
        unique_items = []
        seen = set()
        for item in cleaned_transaction:
            if item not in seen:
                unique_items.append(item)
                seen.add(item)
        
        if unique_items:  # Only add non-empty transactions
            cleaned_transactions.append(unique_items)
    
    logger.info(f"Cleaned {len(transactions)} transactions to {len(cleaned_transactions)}")
    return cleaned_transactions


def calculate_transaction_statistics(transactions: List[List[str]]) -> Dict[str, Any]:
    """Calculate basic statistics for transaction data.
    
    Args:
        transactions: List of transactions.
        
    Returns:
        Dictionary with transaction statistics.
    """
    if not transactions:
        return {}
    
    basket_sizes = [len(transaction) for transaction in transactions]
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    
    # Calculate item frequencies
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    # Calculate co-occurrence matrix
    co_occurrence = {}
    for transaction in transactions:
        for i, item1 in enumerate(transaction):
            for item2 in transaction[i+1:]:
                pair = tuple(sorted([item1, item2]))
                co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
    
    statistics = {
        "total_transactions": len(transactions),
        "total_items": len(all_items),
        "avg_basket_size": np.mean(basket_sizes),
        "min_basket_size": np.min(basket_sizes),
        "max_basket_size": np.max(basket_sizes),
        "std_basket_size": np.std(basket_sizes),
        "most_frequent_items": sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "most_co_occurring_pairs": sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10],
    }
    
    return statistics


def filter_rules_by_metrics(
    rules: pd.DataFrame,
    min_support: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_lift: Optional[float] = None,
    min_conviction: Optional[float] = None,
) -> pd.DataFrame:
    """Filter association rules by metric thresholds.
    
    Args:
        rules: DataFrame with association rules.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        min_lift: Minimum lift threshold.
        min_conviction: Minimum conviction threshold.
        
    Returns:
        Filtered DataFrame with association rules.
    """
    filtered_rules = rules.copy()
    
    if min_support is not None:
        filtered_rules = filtered_rules[filtered_rules["support"] >= min_support]
    
    if min_confidence is not None:
        filtered_rules = filtered_rules[filtered_rules["confidence"] >= min_confidence]
    
    if min_lift is not None:
        filtered_rules = filtered_rules[filtered_rules["lift"] >= min_lift]
    
    if min_conviction is not None:
        filtered_rules = filtered_rules[filtered_rules["conviction"] >= min_conviction]
    
    logger.info(f"Filtered {len(rules)} rules to {len(filtered_rules)} rules")
    return filtered_rules


def calculate_rule_importance_score(
    rules: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Calculate importance scores for association rules.
    
    Args:
        rules: DataFrame with association rules.
        weights: Optional weights for different metrics.
        
    Returns:
        DataFrame with added importance scores.
    """
    if weights is None:
        weights = {
            "support": 0.2,
            "confidence": 0.3,
            "lift": 0.3,
            "conviction": 0.2,
        }
    
    # Normalize metrics to 0-1 scale
    normalized_rules = rules.copy()
    
    for metric in weights.keys():
        if metric in rules.columns:
            min_val = rules[metric].min()
            max_val = rules[metric].max()
            if max_val > min_val:
                normalized_rules[f"{metric}_norm"] = (rules[metric] - min_val) / (max_val - min_val)
            else:
                normalized_rules[f"{metric}_norm"] = 0.5
    
    # Calculate weighted importance score
    importance_scores = []
    for _, rule in normalized_rules.iterrows():
        score = 0
        for metric, weight in weights.items():
            if f"{metric}_norm" in normalized_rules.columns:
                score += weight * rule[f"{metric}_norm"]
        importance_scores.append(score)
    
    normalized_rules["importance_score"] = importance_scores
    
    return normalized_rules


def generate_rule_summary(rules: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary of association rules.
    
    Args:
        rules: DataFrame with association rules.
        
    Returns:
        Dictionary with rule summary statistics.
    """
    if len(rules) == 0:
        return {"error": "No rules to summarize"}
    
    summary = {
        "total_rules": len(rules),
        "avg_support": rules["support"].mean(),
        "avg_confidence": rules["confidence"].mean(),
        "avg_lift": rules["lift"].mean(),
        "avg_conviction": rules["conviction"].mean(),
        "max_support": rules["support"].max(),
        "max_confidence": rules["confidence"].max(),
        "max_lift": rules["lift"].max(),
        "max_conviction": rules["conviction"].max(),
        "min_support": rules["support"].min(),
        "min_confidence": rules["confidence"].min(),
        "min_lift": rules["lift"].min(),
        "min_conviction": rules["conviction"].min(),
    }
    
    # Rule length distribution
    rule_lengths = rules["antecedents"].apply(len) + rules["consequents"].apply(len)
    summary["avg_rule_length"] = rule_lengths.mean()
    summary["rule_length_distribution"] = rule_lengths.value_counts().to_dict()
    
    # Quality categories
    high_quality_rules = rules[
        (rules["confidence"] >= 0.8) & (rules["lift"] >= 2.0)
    ]
    summary["high_quality_rules"] = len(high_quality_rules)
    
    medium_quality_rules = rules[
        (rules["confidence"] >= 0.6) & (rules["lift"] >= 1.5)
    ]
    summary["medium_quality_rules"] = len(medium_quality_rules)
    
    low_quality_rules = rules[
        (rules["confidence"] < 0.6) | (rules["lift"] < 1.5)
    ]
    summary["low_quality_rules"] = len(low_quality_rules)
    
    return summary


def export_rules_to_excel(
    rules: pd.DataFrame,
    output_path: str,
    sheet_name: str = "Association Rules",
) -> None:
    """Export association rules to Excel file with formatting.
    
    Args:
        rules: DataFrame with association rules.
        output_path: Path to save Excel file.
        sheet_name: Name of the Excel sheet.
    """
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Format rules for Excel
            excel_rules = rules.copy()
            excel_rules["antecedents"] = excel_rules["antecedents"].apply(
                lambda x: ", ".join(list(x))
            )
            excel_rules["consequents"] = excel_rules["consequents"].apply(
                lambda x: ", ".join(list(x))
            )
            
            # Write to Excel
            excel_rules.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get workbook and worksheet for formatting
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        logger.info(f"Rules exported to {output_path}")
    
    except Exception as e:
        logger.error(f"Error exporting rules to Excel: {e}")


def create_itemset_network(
    rules: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    """Create network representation of itemsets and rules.
    
    Args:
        rules: DataFrame with association rules.
        min_support: Minimum support for filtering.
        min_confidence: Minimum confidence for filtering.
        
    Returns:
        Dictionary with network data.
    """
    # Filter rules
    filtered_rules = rules[
        (rules["support"] >= min_support) &
        (rules["confidence"] >= min_confidence)
    ]
    
    if len(filtered_rules) == 0:
        return {"error": "No rules meet the criteria"}
    
    # Extract nodes (items) and edges (rules)
    nodes = set()
    edges = []
    
    for _, rule in filtered_rules.iterrows():
        antecedents = list(rule["antecedents"])
        consequents = list(rule["consequents"])
        
        # Add nodes
        nodes.update(antecedents)
        nodes.update(consequents)
        
        # Add edges
        for antecedent in antecedents:
            for consequent in consequents:
                edges.append({
                    "source": antecedent,
                    "target": consequent,
                    "support": rule["support"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                })
    
    # Calculate node statistics
    node_stats = {}
    for node in nodes:
        incoming_edges = [e for e in edges if e["target"] == node]
        outgoing_edges = [e for e in edges if e["source"] == node]
        
        node_stats[node] = {
            "in_degree": len(incoming_edges),
            "out_degree": len(outgoing_edges),
            "total_degree": len(incoming_edges) + len(outgoing_edges),
            "avg_in_confidence": np.mean([e["confidence"] for e in incoming_edges]) if incoming_edges else 0,
            "avg_out_confidence": np.mean([e["confidence"] for e in outgoing_edges]) if outgoing_edges else 0,
        }
    
    network_data = {
        "nodes": list(nodes),
        "edges": edges,
        "node_stats": node_stats,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }
    
    return network_data


def calculate_market_basket_kpis(
    rules: pd.DataFrame,
    transactions: List[List[str]],
    catalog_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Calculate market basket specific KPIs.
    
    Args:
        rules: DataFrame with association rules.
        transactions: List of transactions.
        catalog_df: Optional catalog DataFrame.
        
    Returns:
        Dictionary with market basket KPIs.
    """
    kpis = {}
    
    # Cross-selling potential
    cross_sell_rules = rules[rules["lift"] >= 1.5]
    kpis["cross_sell_potential"] = {
        "total_opportunities": len(cross_sell_rules),
        "avg_lift": cross_sell_rules["lift"].mean() if len(cross_sell_rules) > 0 else 0,
        "high_potential_rules": len(cross_sell_rules[cross_sell_rules["lift"] >= 2.0]),
    }
    
    # Inventory optimization
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    
    kpis["inventory_optimization"] = {
        "total_items": len(all_items),
        "items_in_rules": len(set().union(*[set(rule["antecedents"]) | set(rule["consequents"]) for _, rule in rules.iterrows()])),
        "coverage_percentage": len(set().union(*[set(rule["antecedents"]) | set(rule["consequents"]) for _, rule in rules.iterrows()])) / len(all_items) * 100,
    }
    
    # Store layout optimization
    layout_rules = rules[rules["support"] >= 0.05]
    kpis["layout_optimization"] = {
        "layout_candidates": len(layout_rules),
        "avg_support": layout_rules["support"].mean() if len(layout_rules) > 0 else 0,
        "strong_associations": len(layout_rules[layout_rules["confidence"] >= 0.7]),
    }
    
    # Promotion effectiveness
    promotion_rules = rules[
        (rules["lift"] >= 1.5) &
        (rules["support"] >= 0.02) &
        (rules["confidence"] >= 0.5)
    ]
    kpis["promotion_effectiveness"] = {
        "promotion_candidates": len(promotion_rules),
        "avg_promotion_lift": promotion_rules["lift"].mean() if len(promotion_rules) > 0 else 0,
        "high_impact_rules": len(promotion_rules[promotion_rules["lift"] >= 2.5]),
    }
    
    return kpis
