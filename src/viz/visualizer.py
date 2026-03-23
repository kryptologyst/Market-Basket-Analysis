"""
Market Basket Analysis - Visualization Module

This module provides comprehensive visualization capabilities for market basket
analysis results including interactive charts and business dashboards.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class MarketBasketVisualizer:
    """Creates visualizations for market basket analysis results."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize the visualizer with configuration.
        
        Args:
            config: Configuration dictionary containing visualization settings.
        """
        self.config = config
        self.setup_plotting_style()
    
    def setup_plotting_style(self) -> None:
        """Setup plotting style and configuration."""
        plt.style.use(self.config.visualization.plot_style)
        sns.set_palette("husl")
        
        # Set default figure size and DPI
        plt.rcParams["figure.figsize"] = self.config.visualization.figure_size
        plt.rcParams["figure.dpi"] = self.config.visualization.dpi
    
    def plot_association_rules(
        self,
        association_rules: pd.DataFrame,
        top_n: int = 20,
        metric: str = "lift",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create interactive plot of association rules.
        
        Args:
            association_rules: DataFrame with association rules.
            top_n: Number of top rules to display.
            metric: Metric to sort by ('lift', 'confidence', 'support').
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        if len(association_rules) == 0:
            logger.warning("No association rules to plot")
            return go.Figure()
        
        # Get top N rules
        top_rules = association_rules.nlargest(top_n, metric)
        
        # Create rule labels
        rule_labels = []
        for _, rule in top_rules.iterrows():
            antecedents = ", ".join(list(rule["antecedents"]))
            consequents = ", ".join(list(rule["consequents"]))
            rule_labels.append(f"{antecedents} → {consequents}")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
            subplot_titles=(f"Top {top_n} Association Rules by {metric.title()}",)
        )
        
        # Add bars for the main metric
        fig.add_trace(
            go.Bar(
                x=rule_labels,
                y=top_rules[metric],
                name=metric.title(),
                marker_color=self.config.visualization.colors.primary,
                text=top_rules[metric].round(3),
                textposition="auto",
            ),
            secondary_y=False,
        )
        
        # Add line for confidence
        fig.add_trace(
            go.Scatter(
                x=rule_labels,
                y=top_rules["confidence"],
                name="Confidence",
                mode="lines+markers",
                line=dict(color=self.config.visualization.colors.secondary, width=3),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title=f"Top {top_n} Association Rules by {metric.title()}",
            xaxis_title="Association Rules",
            yaxis_title=f"{metric.title()}",
            yaxis2_title="Confidence",
            height=600,
            showlegend=True,
            xaxis=dict(tickangle=45),
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_support_confidence_lift(
        self,
        association_rules: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create 3D scatter plot of support, confidence, and lift.
        
        Args:
            association_rules: DataFrame with association rules.
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        if len(association_rules) == 0:
            logger.warning("No association rules to plot")
            return go.Figure()
        
        # Create rule labels
        rule_labels = []
        for _, rule in association_rules.iterrows():
            antecedents = ", ".join(list(rule["antecedents"]))
            consequents = ", ".join(list(rule["consequents"]))
            rule_labels.append(f"{antecedents} → {consequents}")
        
        fig = go.Figure(data=go.Scatter3d(
            x=association_rules["support"],
            y=association_rules["confidence"],
            z=association_rules["lift"],
            mode="markers",
            marker=dict(
                size=8,
                color=association_rules["conviction"],
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="Conviction"),
            ),
            text=rule_labels,
            hovertemplate="<b>%{text}</b><br>" +
                         "Support: %{x:.3f}<br>" +
                         "Confidence: %{y:.3f}<br>" +
                         "Lift: %{z:.3f}<br>" +
                         "Conviction: %{marker.color:.3f}<extra></extra>",
        ))
        
        fig.update_layout(
            title="Association Rules: Support vs Confidence vs Lift",
            scene=dict(
                xaxis_title="Support",
                yaxis_title="Confidence",
                zaxis_title="Lift",
            ),
            height=600,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_itemset_support_distribution(
        self,
        frequent_itemsets: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Plot distribution of itemset support values.
        
        Args:
            frequent_itemsets: DataFrame with frequent itemsets.
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        if len(frequent_itemsets) == 0:
            logger.warning("No frequent itemsets to plot")
            return go.Figure()
        
        fig = go.Figure()
        
        # Histogram of support values
        fig.add_trace(go.Histogram(
            x=frequent_itemsets["support"],
            nbinsx=30,
            name="Support Distribution",
            marker_color=self.config.visualization.colors.primary,
            opacity=0.7,
        ))
        
        # Add vertical line for mean
        mean_support = frequent_itemsets["support"].mean()
        fig.add_vline(
            x=mean_support,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_support:.3f}",
        )
        
        fig.update_layout(
            title="Distribution of Itemset Support Values",
            xaxis_title="Support",
            yaxis_title="Frequency",
            height=400,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_rule_length_distribution(
        self,
        association_rules: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Plot distribution of rule lengths.
        
        Args:
            association_rules: DataFrame with association rules.
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        if len(association_rules) == 0:
            logger.warning("No association rules to plot")
            return go.Figure()
        
        # Calculate rule lengths
        rule_lengths = association_rules["antecedents"].apply(len) + \
                      association_rules["consequents"].apply(len)
        
        fig = go.Figure()
        
        # Bar chart of rule length distribution
        length_counts = rule_lengths.value_counts().sort_index()
        
        fig.add_trace(go.Bar(
            x=length_counts.index,
            y=length_counts.values,
            name="Rule Length Distribution",
            marker_color=self.config.visualization.colors.secondary,
        ))
        
        fig.update_layout(
            title="Distribution of Association Rule Lengths",
            xaxis_title="Rule Length (Number of Items)",
            yaxis_title="Number of Rules",
            height=400,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_business_kpis(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create dashboard of business KPIs.
        
        Args:
            evaluation_results: Results from evaluation.
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        # Extract business KPIs
        business_kpis = evaluation_results.get("business_kpis", {})
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Cross-sell Opportunities",
                "Inventory Optimization",
                "Layout Optimization",
                "Promotion Effectiveness"
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Cross-sell opportunities
        cross_sell = business_kpis.get("cross_sell_potential", {})
        fig.add_trace(
            go.Bar(
                x=["Total Opportunities", "Avg Confidence", "Avg Lift"],
                y=[
                    cross_sell.get("total_cross_sell_opportunities", 0),
                    cross_sell.get("avg_cross_sell_confidence", 0),
                    cross_sell.get("avg_cross_sell_lift", 0),
                ],
                name="Cross-sell",
                marker_color=self.config.visualization.colors.primary,
            ),
            row=1, col=1
        )
        
        # Inventory optimization
        inventory = business_kpis.get("inventory_optimization", {})
        fig.add_trace(
            go.Bar(
                x=["Unique Items", "High Co-occurrence", "Clustering Score"],
                y=[
                    inventory.get("total_unique_items", 0),
                    inventory.get("high_co_occurrence_pairs", 0),
                    inventory.get("inventory_clustering_potential", 0),
                ],
                name="Inventory",
                marker_color=self.config.visualization.colors.secondary,
            ),
            row=1, col=2
        )
        
        # Layout optimization
        layout = business_kpis.get("layout_optimization", {})
        fig.add_trace(
            go.Bar(
                x=["Optimization Rules", "Avg Support", "Clusters"],
                y=[
                    layout.get("layout_optimization_rules", 0),
                    layout.get("avg_layout_rule_support", 0),
                    len(layout.get("layout_clusters", [])),
                ],
                name="Layout",
                marker_color=self.config.visualization.colors.accent,
            ),
            row=2, col=1
        )
        
        # Promotion effectiveness
        promotion = business_kpis.get("promotion_effectiveness", {})
        fig.add_trace(
            go.Bar(
                x=["Candidate Rules", "Avg Lift", "High Impact"],
                y=[
                    promotion.get("promotion_candidate_rules", 0),
                    promotion.get("avg_promotion_lift", 0),
                    promotion.get("high_impact_rules", 0),
                ],
                name="Promotion",
                marker_color="orange",
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Business KPIs Dashboard",
            height=600,
            showlegend=False,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_coverage_analysis(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Plot coverage analysis results.
        
        Args:
            evaluation_results: Results from evaluation.
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        coverage = evaluation_results.get("coverage_analysis", {})
        
        if not coverage:
            logger.warning("No coverage analysis data to plot")
            return go.Figure()
        
        # Create pie chart for coverage
        covered_items = coverage.get("covered_items", 0)
        total_items = coverage.get("total_items", 1)
        uncovered_items = total_items - covered_items
        
        fig = go.Figure(data=[go.Pie(
            labels=["Covered Items", "Uncovered Items"],
            values=[covered_items, uncovered_items],
            hole=0.3,
            marker_colors=[self.config.visualization.colors.primary, "lightgray"],
        )])
        
        fig.update_layout(
            title=f"Item Coverage Analysis ({coverage.get('coverage_percentage', 0):.1f}%)",
            height=400,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_novelty_analysis(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Plot novelty analysis results.
        
        Args:
            evaluation_results: Results from evaluation.
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        novelty = evaluation_results.get("novelty_analysis", {})
        
        if not novelty:
            logger.warning("No novelty analysis data to plot")
            return go.Figure()
        
        # Create bar chart for novelty categories
        categories = ["Novel Patterns", "Common Patterns"]
        values = [
            novelty.get("novel_patterns", 0),
            novelty.get("common_patterns", 0),
        ]
        
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            marker_color=[self.config.visualization.colors.accent, "orange"],
        )])
        
        fig.update_layout(
            title="Pattern Novelty Analysis",
            xaxis_title="Pattern Type",
            yaxis_title="Number of Patterns",
            height=400,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(
        self,
        association_rules: pd.DataFrame,
        frequent_itemsets: pd.DataFrame,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            association_rules: DataFrame with association rules.
            frequent_itemsets: DataFrame with frequent itemsets.
            evaluation_results: Results from evaluation.
            save_path: Optional path to save the plot.
            
        Returns:
            Plotly figure object.
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Top Association Rules by Lift",
                "Support vs Confidence vs Lift",
                "Itemset Support Distribution",
                "Rule Length Distribution",
                "Business KPIs",
                "Coverage Analysis"
            ),
            specs=[[{"type": "bar"}, {"type": "scatter3d"}],
                   [{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Top rules
        if len(association_rules) > 0:
            top_rules = association_rules.nlargest(10, "lift")
            rule_labels = [
                f"{', '.join(list(rule['antecedents']))} → {', '.join(list(rule['consequents']))}"
                for _, rule in top_rules.iterrows()
            ]
            
            fig.add_trace(
                go.Bar(
                    x=rule_labels,
                    y=top_rules["lift"],
                    name="Top Rules",
                    marker_color=self.config.visualization.colors.primary,
                ),
                row=1, col=1
            )
        
        # 3D scatter plot
        if len(association_rules) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=association_rules["support"],
                    y=association_rules["confidence"],
                    z=association_rules["lift"],
                    mode="markers",
                    marker=dict(size=5, color=association_rules["conviction"]),
                    name="Rules 3D",
                ),
                row=1, col=2
            )
        
        # Support distribution
        if len(frequent_itemsets) > 0:
            fig.add_trace(
                go.Histogram(
                    x=frequent_itemsets["support"],
                    name="Support Dist",
                    marker_color=self.config.visualization.colors.secondary,
                ),
                row=2, col=1
            )
        
        # Rule length distribution
        if len(association_rules) > 0:
            rule_lengths = association_rules["antecedents"].apply(len) + \
                          association_rules["consequents"].apply(len)
            length_counts = rule_lengths.value_counts().sort_index()
            
            fig.add_trace(
                go.Bar(
                    x=length_counts.index,
                    y=length_counts.values,
                    name="Rule Lengths",
                    marker_color=self.config.visualization.colors.accent,
                ),
                row=2, col=2
            )
        
        # Business KPIs
        business_kpis = evaluation_results.get("business_kpis", {})
        cross_sell = business_kpis.get("cross_sell_potential", {})
        
        fig.add_trace(
            go.Bar(
                x=["Cross-sell", "Inventory", "Layout", "Promotion"],
                y=[
                    cross_sell.get("total_cross_sell_opportunities", 0),
                    business_kpis.get("inventory_optimization", {}).get("total_unique_items", 0),
                    business_kpis.get("layout_optimization", {}).get("layout_optimization_rules", 0),
                    business_kpis.get("promotion_effectiveness", {}).get("promotion_candidate_rules", 0),
                ],
                name="Business KPIs",
                marker_color="orange",
            ),
            row=3, col=1
        )
        
        # Coverage analysis
        coverage = evaluation_results.get("coverage_analysis", {})
        if coverage:
            covered_items = coverage.get("covered_items", 0)
            total_items = coverage.get("total_items", 1)
            uncovered_items = total_items - covered_items
            
            fig.add_trace(
                go.Pie(
                    labels=["Covered", "Uncovered"],
                    values=[covered_items, uncovered_items],
                    name="Coverage",
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title="Market Basket Analysis Comprehensive Dashboard",
            height=1200,
            showlegend=False,
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
