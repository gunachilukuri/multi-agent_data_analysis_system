from crewai import Agent, Task, Crew
from crewai.tools import tool
import pandas as pd
import os

# ========================================
# TOOL: File Validator
# ========================================
@tool("File Validator Tool")
def validate_file(file_path: str) -> str:
    """
    Validates uploaded file for format, readability, and data quality issues.
    Returns detailed validation report.
    
    Args:
        file_path: Path to the file to validate
    """
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"ERROR: File not found at {file_path}"
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Validate file type
        if file_ext not in ['.csv', '.xlsx', '.xls']:
            return f"ERROR: Unsupported file type '{file_ext}'. Only CSV and Excel files accepted."
        
        # Try to read the file
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Validation checks
        issues = []
        
        # Check 1: Empty file
        if df.empty:
            issues.append("File is empty (no data)")
        
        # Check 5: Data type summary
        dtype_summary = df.dtypes.value_counts().to_dict()
        
        # Build validation report
        if not issues:
            status = "VALID"
            report = f"""
VALIDATION STATUS: {status}

FILE INFO:
- Format: {file_ext}
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Names: {list(df.columns)}
- Data Types: {dtype_summary}
- Missing Values: {df.isna().sum().sum()} total cells

✅ File is ready for analysis!
"""
            return report 
        return "Empty file"
        
    except Exception as e:
        return f"ERROR: Failed to read file. Details: {str(e)}"