from crewai.tools import tool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ========================================
# TOOL: Predictive Modeling Tool
# ========================================
@tool("Predictive Modeling Tool")
def build_prediction_model(
    file_path: str, 
    target_column: str, 
    feature_columns: str,
    model_type: str = "auto"
) -> str:
    """
    Builds and evaluates multiple ML models for prediction tasks.
    
    Args:
        file_path: Path to the cleaned data file
        target_column: Column to predict (e.g., 'price')
        feature_columns: Comma-separated list of feature columns (e.g., 'area,bedrooms,bathrooms')
        model_type: Type of model - 'auto', 'linear', 'ridge', 'lasso', 'rf', 'gb', 'dt'
    
    Returns:
        Model performance report with best model recommendation
    """
    
    try:
        # Read data
        df = pd.read_csv(file_path)
        
        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Validate columns exist
        if target_column not in df.columns:
            return f"ERROR: Target column '{target_column}' not found in dataset"
        
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return f"ERROR: Features not found in dataset: {missing_features}"
        
        # Prepare data
        X = df[features].copy()
        y = df[target_column].copy()
        
        # Handle categorical variables (encode them)
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        report = []
        report.append("=" * 70)
        report.append("PREDICTIVE MODELING REPORT")
        report.append("=" * 70)
        report.append(f"\nTarget Variable: {target_column}")
        report.append(f"Features Used: {', '.join(features)}")
        report.append(f"Training Samples: {len(X_train)}")
        report.append(f"Testing Samples: {len(X_test)}")
        
        if categorical_cols.any():
            report.append(f"Categorical Features Encoded: {', '.join(categorical_cols)}")
        
        # Define models to test
        models = {}
        if model_type == "auto":
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=1.0),
                "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            }
        else:
            model_map = {
                "linear": LinearRegression(),
                "ridge": Ridge(alpha=1.0),
                "lasso": Lasso(alpha=1.0),
                "dt": DecisionTreeRegressor(random_state=42, max_depth=10),
                "rf": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                "gb": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            }
            if model_type in model_map:
                models = {model_type.upper(): model_map[model_type]}
            else:
                return f"ERROR: Unknown model type '{model_type}'. Use: auto, linear, ridge, lasso, rf, gb, dt"
        
        # Train and evaluate models
        results = {}
        report.append("\n" + "=" * 70)
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("=" * 70)
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            }
            
            report.append(f"\n📊 {name}:")
            report.append(f"   Train R² Score: {train_r2:.4f}")
            report.append(f"   Test R² Score: {test_r2:.4f}")
            report.append(f"   RMSE: {test_rmse:.2f}")
            report.append(f"   MAE: {test_mae:.2f}")
            
            # Check for overfitting
            if train_r2 - test_r2 > 0.1:
                report.append(f"   ⚠️ Warning: Potential overfitting detected")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        best_model = results[best_model_name]['model']
        best_r2 = results[best_model_name]['test_r2']
        best_rmse = results[best_model_name]['rmse']
        
        report.append("\n" + "=" * 70)
        report.append("BEST MODEL SELECTED")
        report.append("=" * 70)
        report.append(f"\n🏆 Winner: {best_model_name}")
        report.append(f"   R² Score: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)")
        report.append(f"   RMSE: {best_rmse:.2f}")
        report.append(f"   MAE: {results[best_model_name]['mae']:.2f}")
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            report.append("\n📈 FEATURE IMPORTANCE:")
            importances = best_model.feature_importances_
            feature_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            for feat, imp in feature_imp:
                report.append(f"   - {feat}: {imp:.4f} ({imp*100:.2f}%)")
        elif hasattr(best_model, 'coef_'):
            report.append("\n📈 FEATURE COEFFICIENTS:")
            coefs = best_model.coef_
            for feat, coef in zip(features, coefs):
                report.append(f"   - {feat}: {coef:.4f}")
        
        # Save best model
        model_filename = file_path.replace('_cleaned.csv', '_best_model.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        
        report.append(f"\n💾 Best model saved to: {model_filename}")
        
        # Recommendations
        report.append("\n" + "=" * 70)
        report.append("RECOMMENDATIONS")
        report.append("=" * 70)
        
        if best_r2 > 0.8:
            report.append("✅ Excellent model performance! Ready for production use.")
        elif best_r2 > 0.6:
            report.append("✓ Good model performance. Consider feature engineering for improvement.")
        else:
            report.append("⚠️ Model performance is moderate. Consider:")
            report.append("   - Adding more relevant features")
            report.append("   - Feature engineering (interactions, polynomials)")
            report.append("   - Collecting more data")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"ERROR: Modeling failed. Details: {str(e)}"