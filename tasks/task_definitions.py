from crewai import Task
from agents.agent_1_validator import input_validator_agent
from agents.agent_2_problem_analyzer import problem_analyzer_agent
from agents.agent_3_data_cleaner import data_cleaner_agent
from agents.agent_4_data_analyzer import data_analyzer_agent
from agents.agent_5_predictive_modeler import predictive_modeler_agent
from agents.agent_6_insight_generator import insight_agent
from agents.agent_7_web_validator import web_scraper_agent
from agents.agent_8_qa_agent import qa_agent
from agents.agent_9_executive_report import executive_report_agent 

# Task 1: Validate Input File
validate_file_task = Task(
    description="""
    Use the File Validator Tool to validate the input file at: {file_path}
    
    Your responsibilities:
    1. Check if the file exists and is accessible
    2. Verify the file format (CSV or Excel)
    3. Validate data structure and quality
    4. Identify any corruption, missing values, or format issues
    5. Provide a clear VALID or INVALID verdict
    
    Be thorough and detailed in your validation report.
    """,
    
    expected_output="""
    A comprehensive validation report that includes:
    - Validation status (VALID or INVALID)
    - File format and basic statistics
    - List of any issues found
    - Recommendation on whether to proceed with analysis
    """,
    
    agent=input_validator_agent
)

# task 2: Analyze Problem Statement
analyze_problem_task = Task(
    description="""
    Analyze the user's problem statement: {problem_statement}
    And the validated file at: {file_path}
    
    Your responsibilities:
    1. Understand what the user wants to achieve
    2. Identify the type of analysis needed (prediction, classification, etc.)
    3. Examine available columns in the dataset
    4. Determine which columns are likely relevant
    5. Identify potential target variable (if applicable)
    6. Provide clear guidance for the next agents
    
    Use the Problem Statement Analyzer Tool to examine the data and problem.
    """,
    
    expected_output="""
    A detailed analysis report including:
    - Problem statement interpretation
    - Task type (Regression/Classification/Clustering/Analysis)
    - List of available columns
    - Recommended relevant columns for analysis
    - Identified target variable (if ML task)
    - Clear instructions for data cleaning agent
    """,
    
    agent=problem_analyzer_agent
)

# Task 3: Clean Data
clean_data_task = Task(
    description="""
    Clean the validated dataset at: {file_path}
    
    Based on the problem statement analysis, perform comprehensive data cleaning:
    
    1. Remove duplicate rows
    2. Handle missing values appropriately:
       - For numeric columns: fill with median
       - For text columns: fill with mode (most common value)
    3. Drop columns with excessive missing data (>50%)
    4. Clean text data (remove extra whitespace)
    5. Ensure data types are correct
    6. Save the cleaned dataset
    
    Use the Data Cleaning Tool with appropriate parameters.
    Be thorough but preserve data integrity.
    """,
    
    expected_output="""
    A detailed cleaning report that includes:
    - Original vs cleaned dataset dimensions
    - action to be taken Task type (Regression/Classification/Clustering/Analysis)
    - List of all cleaning actions taken
    - Number of duplicates removed
    - Missing values handled (which columns, how many, method used)
    - Columns dropped (if any)
    - Path to the cleaned dataset file
    - Final column list and data types
    """,
    
    agent=data_cleaner_agent
)

# Task 4: Analyze Data
analyze_data_task = Task(
    description="""
    Analyze the cleaned dataset at: {file_path}
    Based on problem statement: {problem_statement}
    
    Steps:
    1. Identify target variable and relevant features from the problem statement
    2. Use the Data Analysis Tool to analyze ONLY relevant columns
    3. Return the COMPLETE analysis report from the tool
    
    IMPORTANT: Return the FULL detailed report from the Data Analysis Tool.
    Do NOT summarize - provide all statistics, correlations, and insights exactly as generated.
    """,
    
    expected_output="""
    The COMPLETE analysis report from the Data Analysis Tool including:
    - Full dataset overview with exact numbers
    - Complete statistical summaries for each numeric column
    - All correlation values
    - All key insights listed
    - All recommendations
    
    Return the tool output verbatim without summarizing.
    """,
    
    agent=data_analyzer_agent,
    context=[validate_file_task, analyze_problem_task, clean_data_task]
)

# Task 5: Build Predictive Model
build_model_task = Task(
    description="""
    Build predictive models based on:
    - Problem statement: {problem_statement}
    - Cleaned data file
    - Analysis insights from previous agent
    
    Steps:
    1. Identify target variable and features from problem statement
    2. Use Predictive Modeling Tool to train multiple models
    3. Compare model performance (R², RMSE, MAE)
    4. Select the best performing model
    5. Analyze feature importance
    6. Save the best model
    7. Provide recommendations
    
    IMPORTANT: Test multiple models and return the COMPLETE performance report.
    Include all metrics, comparisons, and the winning model details.
    """,
    
    expected_output="""
    Complete modeling report including:
    - All models tested with their performance metrics
    - Best model identified with detailed metrics
    - Feature importance analysis
    - Model file path
    - Clear recommendations on model readiness
    
    Return the full tool output without summarization.
    """,
    
    agent=predictive_modeler_agent,
    context=[validate_file_task, analyze_problem_task, clean_data_task, analyze_data_task]
)

# Task 6: Generate Insights
generate_insights_task = Task(
    description="""
    Generate comprehensive business insights and recommendations based on:
    - Problem statement: {problem_statement}
    - Data analysis results
    - Model performance and predictions
    
    Steps:
    1. Summarize the problem and what was achieved
    2. Extract key data insights (patterns, distributions, trends)
    3. Interpret model results in business terms
    4. Identify the most important drivers/factors
    5. Provide clear, actionable business recommendations
    6. Highlight limitations and considerations
    7. Create an executive summary
    
    IMPORTANT: Make insights business-friendly, not technical.
    Focus on "what this means" and "what to do about it".
    Return the COMPLETE insights report.
    """,
    
    expected_output="""
    A comprehensive business insights report including:
    - Problem statement recap
    - Key data insights and patterns
    - Model performance interpretation in business terms
    - Top drivers/factors explained
    - Actionable business recommendations (4-6 specific actions)
    - Limitations and considerations
    - Executive summary
    
    Written for business stakeholders, not data scientists.
    Return the full tool output without summarization.
    """,
    
    agent=insight_agent,
    context=[validate_file_task, analyze_problem_task, clean_data_task, analyze_data_task, build_model_task]
)

# Task 7: Validate Web Data
validate_external_data_task = Task(
    description="""
    Validate the analysis findings against external market data and benchmarks.
    
    Based on:
    - Problem statement: {problem_statement}
    - Internal analysis results
    - Model predictions and performance
    
    Steps:
    1. Identify the problem domain (real estate, finance, healthcare, etc.)
    2. Summarize key findings from internal analysis
    3. Use the Web Data Validator Tool to check against market context
    4. Compare findings with industry benchmarks
    5. Validate model performance against typical accuracy ranges
    6. Identify any discrepancies or concerns
    7. Provide confidence level for the analysis
    8. Suggest external data sources for ongoing validation
    
    IMPORTANT: Provide honest validation - flag any concerns if findings 
    don't align with market reality. Return the COMPLETE validation report.
    """,
    
    expected_output="""
    A comprehensive external validation report including:
    - Summary of internal findings being validated
    - External market context and benchmarks
    - Comparison with industry standards
    - Data quality validation assessment
    - Confidence level (LOW/MEDIUM/HIGH)
    - Recommendations for improvement
    - Suggested external data sources
    - Final validation verdict (VALIDATED / NEEDS REVIEW / REJECTED)
    
    Return the full tool output without summarization.
    """,
    
    agent=web_scraper_agent,
    context=[validate_file_task, analyze_problem_task, clean_data_task, 
             analyze_data_task, build_model_task, generate_insights_task]
)

# Task 8: Perform Quality Assurance
qa_final_check_task = Task(
    description="""
    Perform final comprehensive QA check on the entire pipeline.
    
    Validate:
    - Original file: {file_path}
    - Problem statement: {problem_statement}
    - All generated outputs (cleaned data, model file)
    - Data quality and integrity
    - Model validity and functionality
    - Pipeline completeness
    - Alignment with problem statement
    
    Steps:
    1. Verify all input/output files exist
    2. Check data cleaning quality (duplicates, missing values, anomalies)
    3. Validate data integrity (types, ranges, consistency)
    4. Test model loading and prediction capability
    5. Verify pipeline completeness (all expected outputs generated)
    6. Confirm alignment with original problem statement
    7. Generate comprehensive QA report
    8. Provide final production-readiness verdict
    
    IMPORTANT: Be thorough and critical. Flag ANY issues.
    Return the COMPLETE QA report with pass/fail verdict.
    """,
    
    expected_output="""
    A comprehensive QA report including:
    - File validation status (all files present and accessible)
    - Data cleaning quality assessment (duplicates, missing values)
    - Data integrity check results (types, anomalies)
    - Model validation (exists, loads, predicts correctly)
    - Pipeline completeness verification
    - Problem alignment confirmation
    - Summary of passed/failed checks
    - List of critical issues (if any)
    - List of warnings (if any)
    - Final verdict: PASS / CONDITIONAL PASS / FAIL
    - Production-readiness recommendation
    
    Return the full tool output without summarization.
    """,
    
    agent=qa_agent,
    context=[validate_file_task, analyze_problem_task, clean_data_task, 
             analyze_data_task, build_model_task, generate_insights_task, 
             validate_external_data_task]
)

# Task 9: Generate Executive Report
generate_final_report_task = Task(
    description="""
    Create a CONTEXT-AWARE, USER-INTENT-DRIVEN FINAL REPORT.

    IMPORTANT: This system supports FOUR analysis modes:
    1. ANALYSIS (Descriptive & Diagnostic insights)
    2. PREDICTION (Future outcomes / risk estimation)
    3. CLASSIFICATION (Risk groups / labels)
    4. CLUSTERING (Segment discovery)

    ⚠️ CRITICAL INTENT RULE:
    - Detect what the USER ASKED FOR.
    - Present ONLY the relevant section(s).
    - DO NOT over-deliver unused sections.
    - Prediction/classification may be mentioned briefly ONLY if it adds clarity — never dominate.

    DOMAIN AWARENESS (MANDATORY):
    - If the problem is BUSINESS / MARKET / SALES:
      → Use executive, opportunity-driven, strategy-focused language.
    - If the problem is HEALTH / MEDICAL / RISK TO LIFE:
      → Shift tone to EMPATHETIC, CAUTIOUS, HUMAN-CENTERED.
      → Highlight risk severity clearly.
      → Avoid business framing.
      → Translate statistics into *human impact* (fear, urgency, prevention).

    You have access to:
    - Agent 1: File validation
    - Agent 2: Problem understanding
    - Agent 3: Data cleaning
    - Agent 4: Statistical analysis (KEY INPUT)
    - Agent 5: Models (classification/prediction if used)
    - Agent 6: Insights
    - Agent 7: External validation
    - Agent 8: QA

    YOUR ROLE:
    Transform Agent 4 (and relevant agents) into a FINAL REPORT that:
    - Matches the user’s intent
    - Matches the domain emotion
    - Answers: “What does this mean for ME?”

    -------------------------------------------------
    STRUCTURE (DYNAMIC – USE ONLY WHAT APPLIES)
    -------------------------------------------------

    📋 SECTION 1: EXECUTIVE / PATIENT SUMMARY (Mandatory)
    - What was asked
    - What was analyzed
    - What it means (business value OR health risk)

    📊 SECTION 2: CORE ANALYSIS (Mandatory)
    - Use Agent 4 statistics
    - Explain patterns, distributions, correlations
    - Translate numbers into implications

    📈 SECTION 3: PREDICTION INSIGHTS (ONLY IF USER REQUESTED)
    - What was predicted
    - Confidence level
    - Practical meaning (not technical metrics)

    🏷️ SECTION 4: CLASSIFICATION INSIGHTS (ONLY IF USER REQUESTED)
    - Risk groups / labels
    - What each group means in real life

    🧩 SECTION 5: CLUSTERING INSIGHTS (ONLY IF USER REQUESTED)
    - Natural segments discovered
    - How segments differ
    - Why segmentation matters

    ❤️ HEALTH MODE ADDITIONS (AUTO-ACTIVATE IF MEDICAL):
    - Explicit risk levels (Low / Moderate / High / Critical)
    - Clear warnings where risk is high
    - Preventive or follow-up guidance
    - Empathetic language (human-first, not KPI-first)

    💼 BUSINESS MODE ADDITIONS (AUTO-ACTIVATE IF BUSINESS):
    - Opportunities & gaps
    - Strategic focus areas
    - Market/category prioritization

    🎯 FINAL SECTION: CONCLUSION & NEXT STEPS
    - Clear takeaway
    - What the user should do next
    - Save the report file and confirm download status

    -------------------------------------------------
    ABSOLUTE RULES:
    -------------------------------------------------
    - Never present unused analysis types
    - Never frame health risks like revenue opportunities
    - Every statistic must answer “Why does this matter?”
    - Write for humans, not analysts
    - Be emotionally intelligent

    """,
    
    expected_output="""
    A user-intent-aligned final report that includes ONLY what was requested.

    The output must:
    ✅ Adapt to ANALYSIS / PREDICTION / CLASSIFICATION / CLUSTERING
    ✅ Match domain emotion (Business vs Health)
    ✅ Translate statistics into real-world meaning
    ✅ Avoid unnecessary models or sections
    ✅ Be clear, empathetic, and actionable
    ✅ Save the final report file
    ✅ Explicitly state whether the file was successfully downloaded

    Health reports must:
    - Clearly communicate risk severity
    - Emphasize caution and preventive insight
    - Avoid business-style framing

    Business reports must:
    - Highlight opportunities, gaps, and strategy
    - Focus on decision-making impact

    This is the ONLY output the user sees.
    It must feel intelligent, relevant, and human.

    """,
    
    agent=executive_report_agent,
    context=[
        validate_file_task,
        analyze_problem_task,
        clean_data_task,
        analyze_data_task,              # ← Reads Agent 4's statistics
        build_model_task,
        generate_insights_task,
        validate_external_data_task,
        qa_final_check_task
    ]
)

# Aggregate all tasks into a list for easy access
all_tasks = [
    validate_file_task,
    analyze_problem_task,
    clean_data_task,
    analyze_data_task,
    build_model_task,
    generate_insights_task,
    validate_external_data_task,
    qa_final_check_task,
    generate_final_report_task
]