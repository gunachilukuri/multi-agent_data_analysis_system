from crewai.tools import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ========================================
# TOOL: Data Analysis Tool
# ========================================
@tool("Data Analysis Tool")
def analyze_data(file_path: str, target_column: str = None, relevant_columns: str = None) -> str:
    """
    Performs comprehensive statistical analysis and generates insights from cleaned data.
    
    Args:
        file_path: Path to the cleaned data file
        target_column: The column to predict/analyze (e.g., 'price')
        relevant_columns: Comma-separated list of columns to focus on (e.g., 'area,bedrooms,bathrooms')
    
    Returns:
        Detailed analysis report with statistics and insights
    """
    
    try:
        # Read cleaned data
        df = pd.read_csv(file_path)
        
        # Filter to relevant columns if specified
        if relevant_columns:
            cols_list = [col.strip() for col in relevant_columns.split(',')]
            # Keep target column if specified
            if target_column and target_column not in cols_list:
                cols_list.append(target_column)
            # Only keep columns that exist
            cols_list = [col for col in cols_list if col in df.columns]
            df = df[cols_list]
        
        analysis_report = []
        
        # 1. Basic Statistics
        analysis_report.append("=" * 60)
        analysis_report.append("DATASET OVERVIEW")
        analysis_report.append("=" * 60)
        analysis_report.append(f"Total Rows: {len(df)}")
        analysis_report.append(f"Total Columns: {len(df.columns)}")
        analysis_report.append(f"Columns Analyzed: {', '.join(df.columns)}")
        analysis_report.append("")
        
        # 2. Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            analysis_report.append("=" * 60)
            analysis_report.append("NUMERIC COLUMNS STATISTICS")
            analysis_report.append("=" * 60)
            
            for col in numeric_cols:
                analysis_report.append(f"\n📊 {col.upper()}:")
                analysis_report.append(f"  - Mean: {df[col].mean():.2f}")
                analysis_report.append(f"  - Median: {df[col].median():.2f}")
                analysis_report.append(f"  - Std Dev: {df[col].std():.2f}")
                analysis_report.append(f"  - Min: {df[col].min():.2f}")
                analysis_report.append(f"  - Max: {df[col].max():.2f}")
                analysis_report.append(f"  - 25th Percentile: {df[col].quantile(0.25):.2f}")
                analysis_report.append(f"  - 75th Percentile: {df[col].quantile(0.75):.2f}")
        
        # 3. Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            analysis_report.append("\n" + "=" * 60)
            analysis_report.append("CATEGORICAL COLUMNS ANALYSIS")
            analysis_report.append("=" * 60)
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                analysis_report.append(f"\n📝 {col.upper()}:")
                analysis_report.append(f"  - Unique Values: {df[col].nunique()}")
                analysis_report.append(f"  - Most Common: {value_counts.index[0]} ({value_counts.values[0]} occurrences)")
                if len(value_counts) <= 10:
                    analysis_report.append(f"  - Distribution:")
                    for val, count in value_counts.items():
                        analysis_report.append(f"    • {val}: {count} ({count/len(df)*100:.1f}%)")
        
        # 4. Correlation analysis (if target column specified)
        if target_column and target_column in df.columns and len(numeric_cols) > 1:
            analysis_report.append("\n" + "=" * 60)
            analysis_report.append(f"CORRELATION WITH TARGET: {target_column.upper()}")
            analysis_report.append("=" * 60)
            
            correlations = df[numeric_cols].corr()[target_column].sort_values(ascending=False)
            analysis_report.append("\nTop Correlated Features:")
            for col, corr_val in correlations.items():
                if col != target_column:
                    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.4 else "Weak"
                    direction = "Positive" if corr_val > 0 else "Negative"
                    analysis_report.append(f"  - {col}: {corr_val:.3f} ({strength} {direction})")
        
        # 5. Key Insights
        analysis_report.append("\n" + "=" * 60)
        analysis_report.append("KEY INSIGHTS")
        analysis_report.append("=" * 60)
        
        insights = []
        
        # Insight 1: Target variable distribution
        if target_column and target_column in df.columns:
            target_mean = df[target_column].mean()
            target_median = df[target_column].median()
            if target_mean > target_median * 1.2:
                insights.append(f"✓ {target_column} is right-skewed (mean > median), indicating some high outliers")
            elif target_mean < target_median * 0.8:
                insights.append(f"✓ {target_column} is left-skewed (mean < median), indicating some low outliers")
            else:
                insights.append(f"✓ {target_column} distribution is relatively symmetric")
        
        # Insight 2: Feature relationships
        if target_column and len(numeric_cols) > 1:
            top_corr = correlations.drop(target_column).abs().idxmax()
            top_corr_val = correlations[top_corr]
            insights.append(f"✓ Strongest predictor: '{top_corr}' (correlation: {top_corr_val:.3f})")
        
        # Insight 3: Data quality
        insights.append(f"✓ Dataset contains {len(df)} samples ready for modeling")
        insights.append(f"✓ {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
        
        for insight in insights:
            analysis_report.append(insight)
        
        # 6. Recommendations
        analysis_report.append("\n" + "=" * 60)
        analysis_report.append("RECOMMENDATIONS FOR NEXT STEPS")
        analysis_report.append("=" * 60)
        
        if target_column and len(numeric_cols) > 1:
            analysis_report.append("✓ Data is ready for predictive modeling")
            analysis_report.append(f"✓ Consider using features: {', '.join([c for c in numeric_cols if c != target_column])}")
            if len(categorical_cols) > 0:
                analysis_report.append(f"✓ Encode categorical variables: {', '.join(categorical_cols)}")
        
        return "\n".join(analysis_report)
        
    except Exception as e:
        return f"ERROR: Analysis failed. Details: {str(e)}"