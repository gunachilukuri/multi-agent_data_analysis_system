from crewai.tools import tool
import os
from datetime import datetime

@tool("Report Saver Tool")
def save_report_to_file(report_content: str, file_name: str = None) -> str:
    """
    Saves the executive report to a text file in the outputs/reports directory.
    
    Args:
        report_content: The full report text to save
        file_name: Optional custom filename (default: Executive_Report_YYYY-MM-DD_HHMMSS.txt)
    
    Returns:
        Confirmation message with file path
    """
    try:
        # Create outputs/reports directory if it doesn't exist
        os.makedirs("outputs/reports", exist_ok=True)
        
        # Generate filename if not provided
        if not file_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            file_name = f"Executive_Report_{timestamp}.txt"
        
        # Ensure .txt extension
        if not file_name.endswith('.txt'):
            file_name += '.txt'
        
        # Full path
        file_path = os.path.join("outputs", "reports", file_name)
        
        # Write report
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return f"✅ Report saved successfully to: {file_path}"
        
    except Exception as e:
        return f"❌ Failed to save report: {str(e)}"