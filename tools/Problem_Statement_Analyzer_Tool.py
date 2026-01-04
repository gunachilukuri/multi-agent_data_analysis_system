from crewai.tools import tool
import pandas as pd

# ========================================
# TOOL: Problem Statement Analyzer Tool
# ========================================
@tool("Problem Statement Analyzer Tool")
def analyze_problem_statement(problem_statement: str, file_path: str) -> str:
    """
    Analyzes user's problem statement and identifies relevant columns and task type.
    
    Args:
        problem_statement: User's description of what they want to analyze
        file_path: Path to the data file
    """
    
    try:
        # Read the file to get column names
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        columns = list(df.columns)
        
        # Analyze problem statement for keywords
        statement_lower = problem_statement.lower()
        
        # Detect task type
        task_type = "Unknown"
        if any(word in statement_lower for word in ['predict', 'forecast', 'estimate']):
            task_type = "Regression (Prediction)"
        elif any(word in statement_lower for word in ['classify', 'categorize', 'identify']):
            task_type = "Classification"
        elif any(word in statement_lower for word in ['analyze', 'understand', 'explore', 'insight']):
            task_type = "Exploratory Analysis"
        elif any(word in statement_lower for word in ['cluster', 'segment', 'group']):
            task_type = "Clustering"
        
        # Basic analysis report
        report = f"""
PROBLEM STATEMENT ANALYSIS:

User Query: {problem_statement}

Task Type Detected: {task_type}

Available Columns in Dataset:
{', '.join(columns)}

Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns

Next Steps:
1. Identify target variable from problem statement
2. Select relevant features for analysis
3. Proceed with data cleaning focused on relevant columns
"""
        
        return report
        
    except Exception as e:
        return f"ERROR: Could not analyze problem statement. Details: {str(e)}"