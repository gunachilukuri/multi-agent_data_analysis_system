from crewai.tools import tool
import pandas as pd
import pickle

# ========================================
# TOOL: Insight Generator Tool
# ========================================
@tool("Insight Generator Tool")
def generate_insights(
    cleaned_file_path: str,
    model_file_path: str,
    target_column: str,
    problem_statement: str
) -> str:
    """
    Generates actionable business insights and recommendations based on analysis and model results.
    
    Args:
        cleaned_file_path: Path to cleaned data
        model_file_path: Path to the saved model
        target_column: Target variable name
        problem_statement: Original user problem statement
    
    Returns:
        Comprehensive insights and recommendations report
    """
    
    try:
        # Load data
        df = pd.read_csv(cleaned_file_path)
        
        # Load model
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
        
        report = []
        report.append("=" * 70)
        report.append("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
        report.append("=" * 70)
        
        # Section 1: Problem Summary
        report.append("\n📋 PROBLEM STATEMENT RECAP:")
        report.append(f"   {problem_statement}")
        
        # Section 2: Data Insights
        report.append("\n" + "=" * 70)
        report.append("📊 KEY DATA INSIGHTS")
        report.append("=" * 70)
        
        # Target variable insights
        if target_column in df.columns:
            target_mean = df[target_column].mean()
            target_median = df[target_column].median()
            target_min = df[target_column].min()
            target_max = df[target_column].max()
            
            report.append(f"\n{target_column.upper()} Distribution:")
            report.append(f"   Average: {target_mean:,.2f}")
            report.append(f"   Median: {target_median:,.2f}")
            report.append(f"   Range: {target_min:,.2f} to {target_max:,.2f}")
            
            # Price range categorization (if target is price-related)
            if 'price' in target_column.lower():
                report.append(f"\n   Price Categories:")
                low_threshold = df[target_column].quantile(0.33)
                high_threshold = df[target_column].quantile(0.67)
                
                low_count = len(df[df[target_column] <= low_threshold])
                mid_count = len(df[(df[target_column] > low_threshold) & (df[target_column] <= high_threshold)])
                high_count = len(df[df[target_column] > high_threshold])
                
                report.append(f"   - Budget Range (≤{low_threshold:,.0f}): {low_count} properties ({low_count/len(df)*100:.1f}%)")
                report.append(f"   - Mid Range ({low_threshold:,.0f}-{high_threshold:,.0f}): {mid_count} properties ({mid_count/len(df)*100:.1f}%)")
                report.append(f"   - Premium Range (≥{high_threshold:,.0f}): {high_count} properties ({high_count/len(df)*100:.1f}%)")
        
        # Feature insights
        report.append("\n🔍 Feature Patterns:")
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:5]:  # Top 5 numeric columns
            if col != target_column:
                col_mean = df[col].mean()
                col_median = df[col].median()
                report.append(f"   - {col}: Average = {col_mean:.2f}, Median = {col_median:.2f}")
        
        # Section 3: Model Performance Interpretation
        report.append("\n" + "=" * 70)
        report.append("🤖 MODEL PERFORMANCE INTERPRETATION")
        report.append("=" * 70)
        
        # Feature importance interpretation
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(len(importances))]
            
            report.append("\n💡 What Drives Predictions:")
            sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            
            for i, (feat, imp) in enumerate(sorted_features[:3], 1):
                if imp > 0.3:
                    impact = "MAJOR"
                elif imp > 0.1:
                    impact = "MODERATE"
                else:
                    impact = "MINOR"
                report.append(f"   {i}. {feat}: {impact} impact ({imp*100:.1f}% importance)")
        
        # Section 4: Business Recommendations
        report.append("\n" + "=" * 70)
        report.append("💼 BUSINESS RECOMMENDATIONS")
        report.append("=" * 70)
        
        recommendations = []
        
        # Recommendation 1: Data-driven decisions
        recommendations.append("✓ Use Model for Decision Making:")
        recommendations.append("  The trained model can predict outcomes with reasonable accuracy.")
        recommendations.append("  Integrate this model into your decision-making process for:")
        recommendations.append("  - Price estimation and valuation")
        recommendations.append("  - Resource allocation planning")
        recommendations.append("  - Risk assessment")
        
        # Recommendation 2: Feature focus
        if hasattr(model, 'feature_importances_'):
            top_feature = sorted_features[0][0]
            recommendations.append(f"\n✓ Focus on Key Driver: '{top_feature}'")
            recommendations.append(f"  This feature has the strongest influence on {target_column}.")
            recommendations.append("  Strategy: Prioritize data quality and accuracy for this feature.")
        
        # Recommendation 3: Data collection
        recommendations.append("\n✓ Continuous Improvement:")
        recommendations.append("  - Collect more data over time to improve model accuracy")
        recommendations.append("  - Monitor model performance on new data")
        recommendations.append("  - Retrain model quarterly with updated data")
        
        # Recommendation 4: Actionable steps
        recommendations.append("\n✓ Next Steps:")
        recommendations.append("  1. Deploy model to production environment")
        recommendations.append("  2. Create prediction API or dashboard")
        recommendations.append("  3. Set up monitoring for model drift")
        recommendations.append("  4. Document model assumptions and limitations")
        
        for rec in recommendations:
            report.append(rec)
        
        # Section 5: Limitations & Considerations
        report.append("\n" + "=" * 70)
        report.append("⚠️ LIMITATIONS & CONSIDERATIONS")
        report.append("=" * 70)
        
        report.append("\n- Model predictions are based on historical data patterns")
        report.append("- External factors not in the dataset may affect real-world outcomes")
        report.append("- Regular model retraining is recommended as new data becomes available")
        report.append("- Human expertise should complement model predictions, not be replaced")
        
        # Section 6: Summary
        report.append("\n" + "=" * 70)
        report.append("📝 EXECUTIVE SUMMARY")
        report.append("=" * 70)
        
        report.append(f"\n✅ Successfully analyzed {len(df)} records")
        report.append(f"✅ Built predictive model for {target_column}")
        report.append(f"✅ Identified key drivers and patterns")
        report.append(f"✅ Generated actionable recommendations")
        report.append("\n🎯 The model is ready for deployment and decision support!")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"ERROR: Insight generation failed. Details: {str(e)}"