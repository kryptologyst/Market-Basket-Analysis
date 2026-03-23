"""
Market Basket Analysis - Streamlit Demo Application

Interactive dashboard for market basket analysis with real-time visualization
and business insights generation.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.processor import DataProcessor
from models.basket_analyzer import MarketBasketAnalyzer
from eval.evaluator import MarketBasketEvaluator
from viz.visualizer import MarketBasketVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Market Basket Analysis Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_config() -> DictConfig:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    return OmegaConf.load(config_path)

# Initialize components
@st.cache_resource
def initialize_components(config: DictConfig) -> Tuple[DataProcessor, MarketBasketAnalyzer, MarketBasketEvaluator, MarketBasketVisualizer]:
    """Initialize all analysis components."""
    processor = DataProcessor(config)
    analyzer = MarketBasketAnalyzer(config)
    evaluator = MarketBasketEvaluator(config)
    visualizer = MarketBasketVisualizer(config)
    return processor, analyzer, evaluator, visualizer

def main():
    """Main Streamlit application."""
    # Load configuration and initialize components
    config = load_config()
    processor, analyzer, evaluator, visualizer = initialize_components(config)
    
    # Title and disclaimer
    st.title("🛒 Market Basket Analysis Dashboard")
    
    # Important disclaimer
    st.warning("""
    **IMPORTANT DISCLAIMER**: This is a research and educational tool for market basket analysis. 
    Results should not be used for automated business decisions without human review and validation. 
    Always verify insights with domain experts before implementing any recommendations.
    """)
    
    # Sidebar controls
    st.sidebar.header("Analysis Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Synthetic Data", "Upload Data"],
        help="Choose between synthetic data generation or upload your own transaction data"
    )
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["apriori", "fp_growth", "eclat"],
        help="Choose the association rule mining algorithm"
    )
    
    # Parameter controls
    st.sidebar.subheader("Model Parameters")
    
    min_support = st.sidebar.slider(
        "Minimum Support",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Minimum support threshold for frequent itemsets"
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format="%.2f",
        help="Minimum confidence threshold for association rules"
    )
    
    min_lift = st.sidebar.slider(
        "Minimum Lift",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="Minimum lift threshold for association rules"
    )
    
    max_length = st.sidebar.slider(
        "Maximum Itemset Length",
        min_value=2,
        max_value=10,
        value=5,
        help="Maximum length of itemsets to consider"
    )
    
    # Data generation/loading
    if data_source == "Synthetic Data":
        st.sidebar.subheader("Synthetic Data Parameters")
        
        n_transactions = st.sidebar.number_input(
            "Number of Transactions",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000
        )
        
        n_items = st.sidebar.number_input(
            "Number of Items",
            min_value=50,
            max_value=500,
            value=100,
            step=10
        )
        
        avg_basket_size = st.sidebar.slider(
            "Average Basket Size",
            min_value=2.0,
            max_value=10.0,
            value=4.5,
            step=0.1
        )
        
        # Generate synthetic data
        if st.sidebar.button("Generate Data"):
            with st.spinner("Generating synthetic data..."):
                transactions, catalog_df, customers_df = processor.generate_synthetic_data(
                    n_transactions=n_transactions,
                    n_items=n_items,
                    avg_basket_size=avg_basket_size
                )
                
                # Store in session state
                st.session_state.transactions = transactions
                st.session_state.catalog_df = catalog_df
                st.session_state.customers_df = customers_df
                st.session_state.data_generated = True
    
    else:  # Upload Data
        st.sidebar.subheader("Upload Transaction Data")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload a CSV file with transaction data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                st.session_state.data_generated = True
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
    
    # Analysis execution
    if st.sidebar.button("Run Analysis") and st.session_state.get("data_generated", False):
        with st.spinner("Running market basket analysis..."):
            try:
                # Get data
                if data_source == "Synthetic Data":
                    transactions = st.session_state.transactions
                    catalog_df = st.session_state.catalog_df
                else:
                    # Convert uploaded data to transaction format
                    df = st.session_state.uploaded_df
                    transactions = []
                    for _, group in df.groupby("transaction_id"):
                        items = group["item_name"].tolist()
                        transactions.append(items)
                    catalog_df = pd.DataFrame()  # No catalog for uploaded data
                
                # Run analysis
                analyzer.fit(
                    transactions=transactions,
                    algorithm=algorithm,
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift,
                    max_length=max_length
                )
                
                # Get results
                frequent_itemsets = analyzer.get_frequent_itemsets()
                association_rules = analyzer.get_association_rules()
                
                # Run evaluation
                evaluation_results = evaluator.evaluate(
                    association_rules=association_rules,
                    frequent_itemsets=frequent_itemsets,
                    transactions=transactions,
                    catalog_df=catalog_df
                )
                
                # Store results in session state
                st.session_state.frequent_itemsets = frequent_itemsets
                st.session_state.association_rules = association_rules
                st.session_state.evaluation_results = evaluation_results
                st.session_state.analysis_complete = True
                
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                logger.error(f"Analysis error: {e}")
    
    # Display results
    if st.session_state.get("analysis_complete", False):
        display_results(visualizer, st.session_state)
    
    # Data preview
    if st.session_state.get("data_generated", False):
        display_data_preview(st.session_state)

def display_results(visualizer: MarketBasketVisualizer, session_state: Dict[str, Any]) -> None:
    """Display analysis results."""
    st.header("Analysis Results")
    
    # Get data
    association_rules = session_state.association_rules
    frequent_itemsets = session_state.frequent_itemsets
    evaluation_results = session_state.evaluation_results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Rules",
            len(association_rules),
            help="Number of association rules generated"
        )
    
    with col2:
        st.metric(
            "Frequent Itemsets",
            len(frequent_itemsets),
            help="Number of frequent itemsets found"
        )
    
    with col3:
        avg_confidence = association_rules["confidence"].mean() if len(association_rules) > 0 else 0
        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.3f}",
            help="Average confidence of association rules"
        )
    
    with col4:
        avg_lift = association_rules["lift"].mean() if len(association_rules) > 0 else 0
        st.metric(
            "Avg Lift",
            f"{avg_lift:.2f}",
            help="Average lift of association rules"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Association Rules", "Visualizations", "Business Insights", "Evaluation", "Recommendations"
    ])
    
    with tab1:
        display_association_rules(association_rules)
    
    with tab2:
        display_visualizations(visualizer, association_rules, frequent_itemsets, evaluation_results)
    
    with tab3:
        display_business_insights(evaluation_results)
    
    with tab4:
        display_evaluation_metrics(evaluation_results)
    
    with tab5:
        display_recommendations(association_rules)

def display_association_rules(association_rules: pd.DataFrame) -> None:
    """Display association rules in a table."""
    st.subheader("Association Rules")
    
    if len(association_rules) == 0:
        st.warning("No association rules found with the current parameters.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence_filter = st.slider(
            "Filter by Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
    
    with col2:
        min_lift_filter = st.slider(
            "Filter by Lift",
            min_value=1.0,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    
    # Apply filters
    filtered_rules = association_rules[
        (association_rules["confidence"] >= min_confidence_filter) &
        (association_rules["lift"] >= min_lift_filter)
    ]
    
    st.write(f"Showing {len(filtered_rules)} rules (filtered from {len(association_rules)} total)")
    
    # Display rules
    if len(filtered_rules) > 0:
        # Format rules for display
        display_rules = filtered_rules.copy()
        display_rules["antecedents"] = display_rules["antecedents"].apply(
            lambda x: ", ".join(list(x))
        )
        display_rules["consequents"] = display_rules["consequents"].apply(
            lambda x: ", ".join(list(x))
        )
        
        # Select columns to display
        columns_to_show = ["antecedents", "consequents", "support", "confidence", "lift", "conviction"]
        st.dataframe(
            display_rules[columns_to_show],
            use_container_width=True,
            height=400
        )
    else:
        st.info("No rules match the current filters.")

def display_visualizations(
    visualizer: MarketBasketVisualizer,
    association_rules: pd.DataFrame,
    frequent_itemsets: pd.DataFrame,
    evaluation_results: Dict[str, Any]
) -> None:
    """Display interactive visualizations."""
    st.subheader("Interactive Visualizations")
    
    if len(association_rules) == 0:
        st.warning("No association rules to visualize.")
        return
    
    # Visualization options
    viz_type = st.selectbox(
        "Visualization Type",
        [
            "Top Association Rules",
            "3D Scatter Plot",
            "Itemset Support Distribution",
            "Rule Length Distribution",
            "Business KPIs Dashboard",
            "Coverage Analysis",
            "Comprehensive Dashboard"
        ]
    )
    
    if viz_type == "Top Association Rules":
        top_n = st.slider("Number of Top Rules", 5, 50, 20)
        metric = st.selectbox("Sort by Metric", ["lift", "confidence", "support"])
        
        fig = visualizer.plot_association_rules(association_rules, top_n, metric)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D Scatter Plot":
        fig = visualizer.plot_support_confidence_lift(association_rules)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Itemset Support Distribution":
        fig = visualizer.plot_itemset_support_distribution(frequent_itemsets)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Rule Length Distribution":
        fig = visualizer.plot_rule_length_distribution(association_rules)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Business KPIs Dashboard":
        fig = visualizer.plot_business_kpis(evaluation_results)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Coverage Analysis":
        fig = visualizer.plot_coverage_analysis(evaluation_results)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Comprehensive Dashboard":
        fig = visualizer.create_comprehensive_dashboard(
            association_rules, frequent_itemsets, evaluation_results
        )
        st.plotly_chart(fig, use_container_width=True)

def display_business_insights(evaluation_results: Dict[str, Any]) -> None:
    """Display business insights and KPIs."""
    st.subheader("Business Insights")
    
    business_kpis = evaluation_results.get("business_kpis", {})
    
    if not business_kpis:
        st.warning("No business insights available.")
        return
    
    # Cross-sell opportunities
    st.subheader("Cross-sell Opportunities")
    cross_sell = business_kpis.get("cross_sell_potential", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Opportunities", cross_sell.get("total_cross_sell_opportunities", 0))
    with col2:
        st.metric("Avg Confidence", f"{cross_sell.get('avg_cross_sell_confidence', 0):.3f}")
    with col3:
        st.metric("Avg Lift", f"{cross_sell.get('avg_cross_sell_lift', 0):.2f}")
    
    # Top cross-sell items
    top_items = cross_sell.get("top_cross_sell_items", [])
    if top_items:
        st.write("**Top Cross-sell Items:**")
        for item, count in top_items[:10]:
            st.write(f"- {item}: {count} opportunities")
    
    # Inventory optimization
    st.subheader("Inventory Optimization")
    inventory = business_kpis.get("inventory_optimization", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Items", inventory.get("total_unique_items", 0))
    with col2:
        st.metric("High Co-occurrence Pairs", inventory.get("high_co_occurrence_pairs", 0))
    with col3:
        st.metric("Clustering Score", f"{inventory.get('inventory_clustering_potential', 0):.3f}")
    
    # Layout optimization
    st.subheader("Store Layout Optimization")
    layout = business_kpis.get("layout_optimization", {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimization Rules", layout.get("layout_optimization_rules", 0))
    with col2:
        st.metric("Avg Rule Support", f"{layout.get('avg_layout_rule_support', 0):.3f}")
    
    # Promotion effectiveness
    st.subheader("Promotion Effectiveness")
    promotion = business_kpis.get("promotion_effectiveness", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Candidate Rules", promotion.get("promotion_candidate_rules", 0))
    with col2:
        st.metric("Avg Lift", f"{promotion.get('avg_promotion_lift', 0):.2f}")
    with col3:
        st.metric("High Impact Rules", promotion.get("high_impact_rules", 0))

def display_evaluation_metrics(evaluation_results: Dict[str, Any]) -> None:
    """Display evaluation metrics."""
    st.subheader("Evaluation Metrics")
    
    # ML Metrics
    st.subheader("Machine Learning Metrics")
    ml_metrics = evaluation_results.get("ml_metrics", {})
    
    if ml_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rules", ml_metrics.get("total_rules", 0))
        with col2:
            st.metric("High Confidence Rules", ml_metrics.get("high_confidence_rules", 0))
        with col3:
            st.metric("High Lift Rules", ml_metrics.get("high_lift_rules", 0))
        with col4:
            st.metric("Strong Rules", ml_metrics.get("strong_rules", 0))
        
        # Additional metrics
        st.write("**Detailed Metrics:**")
        metrics_df = pd.DataFrame([
            {"Metric": "Average Support", "Value": f"{ml_metrics.get('avg_support', 0):.4f}"},
            {"Metric": "Average Confidence", "Value": f"{ml_metrics.get('avg_confidence', 0):.4f}"},
            {"Metric": "Average Lift", "Value": f"{ml_metrics.get('avg_lift', 0):.4f}"},
            {"Metric": "Average Conviction", "Value": f"{ml_metrics.get('avg_conviction', 0):.4f}"},
            {"Metric": "Max Lift", "Value": f"{ml_metrics.get('max_lift', 0):.4f}"},
        ])
        st.dataframe(metrics_df, use_container_width=True)
    
    # Coverage analysis
    st.subheader("Coverage Analysis")
    coverage = evaluation_results.get("coverage_analysis", {})
    
    if coverage:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", coverage.get("total_items", 0))
        with col2:
            st.metric("Covered Items", coverage.get("covered_items", 0))
        with col3:
            st.metric("Coverage %", f"{coverage.get('coverage_percentage', 0):.1f}%")
    
    # Novelty analysis
    st.subheader("Novelty Analysis")
    novelty = evaluation_results.get("novelty_analysis", {})
    
    if novelty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Novel Patterns", novelty.get("novel_patterns", 0))
        with col2:
            st.metric("Common Patterns", novelty.get("common_patterns", 0))
        with col3:
            st.metric("Avg Pattern Frequency", f"{novelty.get('avg_pattern_frequency', 0):.1f}")

def display_recommendations(association_rules: pd.DataFrame) -> None:
    """Display recommendation interface."""
    st.subheader("Product Recommendations")
    
    if len(association_rules) == 0:
        st.warning("No association rules available for recommendations.")
        return
    
    # Recommendation interface
    st.write("Enter items in your current basket to get recommendations:")
    
    # Get unique items from rules
    all_items = set()
    for _, rule in association_rules.iterrows():
        all_items.update(rule["antecedents"])
        all_items.update(rule["consequents"])
    
    # Multi-select for basket items
    basket_items = st.multiselect(
        "Select items in your basket:",
        options=sorted(list(all_items)),
        help="Choose items that are currently in your basket"
    )
    
    # Recommendation parameters
    col1, col2 = st.columns(2)
    with col1:
        n_recommendations = st.slider("Number of recommendations", 1, 10, 5)
    with col2:
        min_confidence = st.slider("Minimum confidence", 0.1, 1.0, 0.5, 0.05)
    
    # Generate recommendations
    if st.button("Get Recommendations") and basket_items:
        # Create analyzer instance for recommendations
        config = load_config()
        analyzer = MarketBasketAnalyzer(config)
        
        # Get recommendations
        recommendations = analyzer.predict_recommendations(
            basket=basket_items,
            n_recommendations=n_recommendations,
            min_confidence=min_confidence
        )
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations!")
            
            for i, (item, confidence, rule) in enumerate(recommendations, 1):
                with st.expander(f"Recommendation {i}: {item}"):
                    st.write(f"**Confidence:** {confidence:.3f}")
                    st.write(f"**Rule:** {rule}")
        else:
            st.info("No recommendations found with the current parameters.")
    
    elif basket_items:
        st.info("Click 'Get Recommendations' to see suggestions.")

def display_data_preview(session_state: Dict[str, Any]) -> None:
    """Display data preview."""
    st.header("Data Preview")
    
    if "transactions" in session_state:
        st.subheader("Transaction Data")
        transactions = session_state.transactions
        
        # Show transaction statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(transactions))
        with col2:
            avg_basket_size = sum(len(t) for t in transactions) / len(transactions)
            st.metric("Avg Basket Size", f"{avg_basket_size:.1f}")
        with col3:
            all_items = set()
            for t in transactions:
                all_items.update(t)
            st.metric("Unique Items", len(all_items))
        
        # Show sample transactions
        st.write("**Sample Transactions:**")
        sample_transactions = transactions[:10]
        for i, transaction in enumerate(sample_transactions, 1):
            st.write(f"{i}. {', '.join(transaction)}")
    
    if "catalog_df" in session_state and len(session_state.catalog_df) > 0:
        st.subheader("Product Catalog")
        st.dataframe(session_state.catalog_df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
