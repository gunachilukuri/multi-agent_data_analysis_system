from crewai.tools import tool
import pandas as pd
import numpy as np

# ========================================
# TOOL: Data Cleaning Tool
# ========================================
@tool("Data Cleaning Tool")
def clean_data(file_path: str, missing_threshold: int = 50) -> str:
    """
    Cleans data: removes duplicates, handles missing values, fixes data types.
    
    Args:
        file_path: Path to the file
        missing_threshold: Drop columns with more than this % missing (default 50%)
    
    Returns:
        Report of cleaning actions taken and path to cleaned file
    """
    
    try:
        # 1. Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        original_shape = df.shape
        cleaning_report = []
        
        # 2. Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            cleaning_report.append(f"Removed {duplicates} duplicate rows")
        
        # 3. Drop columns with too much missing data
        missing_pct = (df.isna().sum() / len(df) * 100)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            cleaning_report.append(f"Dropped columns with >{missing_threshold}% missing: {cols_to_drop}")
        
        # 4. Fill missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                # For numeric columns: fill with median
                if df[col].dtype in ['int64', 'float64']:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    cleaning_report.append(f"Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
                
                # For text columns: fill with mode (most common)
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    cleaning_report.append(f"Filled {missing_count} missing values in '{col}' with mode ('{mode_val}')")
        
        # 5. Clean text columns (remove extra spaces)
        text_cols_cleaned = 0
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            text_cols_cleaned += 1
        if text_cols_cleaned > 0:
            cleaning_report.append(f"Cleaned whitespace in {text_cols_cleaned} text columns")
        
        # 6. Save cleaned data
        cleaned_file_path = file_path.replace('.csv', '_cleaned.csv').replace('.xlsx', '_cleaned.csv')
        df.to_csv(cleaned_file_path, index=False)
        
        # 7. Generate report
        report = f"""
DATA CLEANING COMPLETE ✅

Original Shape: {original_shape[0]} rows × {original_shape[1]} columns
Cleaned Shape: {df.shape[0]} rows × {df.shape[1]} columns

Actions Taken:
{chr(10).join(f'- {action}' for action in cleaning_report) if cleaning_report else '- No cleaning needed (data was already clean)'}

Cleaned file saved to: {cleaned_file_path}

Remaining Columns ({len(df.columns)}):
{', '.join(df.columns)}

Data Types:
{df.dtypes.to_dict()}
"""
        
        return report
        
    except Exception as e:
        return f"ERROR: Cleaning failed. Details: {str(e)}"