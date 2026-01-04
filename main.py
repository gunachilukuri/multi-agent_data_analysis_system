import argparse
from crew.data_analyzer_crew import DataAnalyzerCrew

def main():
    parser = argparse.ArgumentParser(description="AI Data Analyzer")
    parser.add_argument("--file", required=True, help="Path to data file")
    parser.add_argument("--problem", required=True, help="Problem statement")
    
    args = parser.parse_args()
    
    print("🚀 Starting AI Data Analyzer...")
    
    crew = DataAnalyzerCrew()
    result = crew.analyze(
        file_path=args.file,
        problem_statement=args.problem
    )
    
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*80)
    print(result)

if __name__ == "__main__":
    main()