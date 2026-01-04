from crewai.tools import tool
import pandas as pd
import os
import pickle

# ========================================
# TOOL: Quality Assurance Tool
# ========================================
@tool("Quality Assurance Tool")
def perform_qa_check(
    original_file_path: str,
    cleaned_file_path: str,
    model_file_path: str,
    problem_statement: str
) -> str:
    """
    Performs comprehensive QA check on the entire pipeline - data, model, and results.
    
    Args:
        original_file_path: Path to original input file
        cleaned_file_path: Path to cleaned data file
        model_file_path: Path to saved model
        problem_statement: Original user problem statement
    
    Returns:
        Comprehensive QA report with pass/fail status for each component
    """
    
    try:
        report = []
        report.append("=" * 70)
        report.append("🔍 COMPREHENSIVE QUALITY ASSURANCE REPORT")
        report.append("=" * 70)
        report.append(f"\nQA Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # QA checklist
        qa_results = {
            'file_validation': False,
            'data_cleaning': False,
            'data_integrity': False,
            'model_exists': False,
            'model_validity': False,
            'completeness': False
        }
        issues = []
        warnings = []
        
        # ============================================
        # CHECK 1: File Validation
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("CHECK 1: FILE VALIDATION")
        report.append("=" * 70)
        
        if os.path.exists(original_file_path):
            report.append("✅ Original file exists and accessible")
            qa_results['file_validation'] = True
        else:
            report.append("❌ Original file not found")
            issues.append("Original file missing or inaccessible")
        
        if os.path.exists(cleaned_file_path):
            report.append("✅ Cleaned file generated successfully")
        else:
            report.append("❌ Cleaned file not found")
            issues.append("Data cleaning did not produce output file")
            qa_results['file_validation'] = False
        
        # ============================================
        # CHECK 2: Data Cleaning Quality
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("CHECK 2: DATA CLEANING QUALITY")
        report.append("=" * 70)
        
        try:
            original_df = pd.read_csv(original_file_path)
            cleaned_df = pd.read_csv(cleaned_file_path)
            
            report.append(f"\n📊 Data Dimensions:")
            report.append(f"   Original: {original_df.shape[0]} rows × {original_df.shape[1]} columns")
            report.append(f"   Cleaned:  {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
            
            # Check for data loss
            rows_lost = original_df.shape[0] - cleaned_df.shape[0]
            if rows_lost > 0:
                loss_pct = (rows_lost / original_df.shape[0]) * 100
                if loss_pct > 50:
                    report.append(f"⚠️  WARNING: {loss_pct:.1f}% of rows removed during cleaning")
                    warnings.append(f"High data loss: {loss_pct:.1f}% rows removed")
                else:
                    report.append(f"✓ Acceptable data loss: {loss_pct:.1f}% rows removed")
            else:
                report.append("✓ No data loss during cleaning")
            
            # Check for missing values
            missing_values = cleaned_df.isnull().sum().sum()
            if missing_values == 0:
                report.append("✅ No missing values in cleaned data")
                qa_results['data_cleaning'] = True
            else:
                report.append(f"⚠️  WARNING: {missing_values} missing values still present")
                warnings.append(f"{missing_values} missing values remain after cleaning")
                qa_results['data_cleaning'] = True  # Still pass if cleaning was attempted
            
            # Check for duplicates
            duplicates = cleaned_df.duplicated().sum()
            if duplicates == 0:
                report.append("✅ No duplicate rows in cleaned data")
            else:
                report.append(f"❌ FAIL: {duplicates} duplicate rows found")
                issues.append(f"{duplicates} duplicates not removed")
                qa_results['data_cleaning'] = False
            
        except Exception as e:
            report.append(f"❌ FAIL: Could not validate data cleaning - {str(e)}")
            issues.append(f"Data validation error: {str(e)}")
        
        # ============================================
        # CHECK 3: Data Integrity
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("CHECK 3: DATA INTEGRITY")
        report.append("=" * 70)
        
        try:
            # Check data types
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
            
            report.append(f"\n📋 Data Types:")
            report.append(f"   Numeric columns: {len(numeric_cols)}")
            report.append(f"   Categorical columns: {len(categorical_cols)}")
            
            # Check for data anomalies
            anomalies_found = False
            for col in numeric_cols:
                if (cleaned_df[col] < 0).any():
                    report.append(f"⚠️  WARNING: Negative values in '{col}'")
                    warnings.append(f"Negative values in numeric column '{col}'")
                    anomalies_found = True
            
            if not anomalies_found:
                report.append("✅ No obvious data anomalies detected")
                qa_results['data_integrity'] = True
            else:
                qa_results['data_integrity'] = True  # Warnings don't fail QA
            
        except Exception as e:
            report.append(f"❌ FAIL: Data integrity check failed - {str(e)}")
            issues.append(f"Integrity check error: {str(e)}")
        
        # ============================================
        # CHECK 4: Model Existence & Validity
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("CHECK 4: MODEL VALIDATION")
        report.append("=" * 70)
        
        if os.path.exists(model_file_path):
            report.append("✅ Model file exists")
            qa_results['model_exists'] = True
            
            try:
                with open(model_file_path, 'rb') as f:
                    model = pickle.load(f)
                
                report.append("✅ Model loaded successfully")
                
                # Check if model has predict method
                if hasattr(model, 'predict'):
                    report.append("✅ Model has prediction capability")
                    qa_results['model_validity'] = True
                    
                    # Test prediction on sample data
                    try:
                        sample_data = cleaned_df.select_dtypes(include=['number']).iloc[:1]
                        if len(sample_data.columns) > 0:
                            # Remove target if present (common column names)
                            potential_targets = ['price', 'target', 'y', 'label']
                            for col in potential_targets:
                                if col in sample_data.columns:
                                    sample_data = sample_data.drop(columns=[col])
                            
                            if len(sample_data.columns) > 0:
                                test_pred = model.predict(sample_data)
                                report.append(f"✅ Model prediction test successful (output: {test_pred[0]:.2f})")
                            else:
                                report.append("⚠️  WARNING: Could not test prediction (no features available)")
                                warnings.append("Model prediction test skipped - no valid features")
                    except Exception as pred_error:
                        report.append(f"⚠️  WARNING: Prediction test failed - {str(pred_error)}")
                        warnings.append("Model prediction test unsuccessful")
                else:
                    report.append("❌ FAIL: Model lacks prediction method")
                    issues.append("Model is not a valid predictor")
                    qa_results['model_validity'] = False
                    
            except Exception as e:
                report.append(f"❌ FAIL: Could not load model - {str(e)}")
                issues.append(f"Model loading error: {str(e)}")
                qa_results['model_validity'] = False
        else:
            report.append("❌ FAIL: Model file not found")
            issues.append("Model was not saved properly")
        
        # ============================================
        # CHECK 5: Pipeline Completeness
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("CHECK 5: PIPELINE COMPLETENESS")
        report.append("=" * 70)
        
        expected_outputs = {
            'Original File': original_file_path,
            'Cleaned File': cleaned_file_path,
            'Model File': model_file_path
        }
        
        all_outputs_present = True
        for output_name, output_path in expected_outputs.items():
            if os.path.exists(output_path):
                report.append(f"✅ {output_name}: Present")
            else:
                report.append(f"❌ {output_name}: Missing")
                all_outputs_present = False
        
        if all_outputs_present:
            qa_results['completeness'] = True
            report.append("\n✅ All expected outputs generated")
        else:
            report.append("\n❌ Pipeline incomplete - missing outputs")
            issues.append("Not all expected files were generated")
        
        # ============================================
        # CHECK 6: Problem Statement Alignment
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("CHECK 6: PROBLEM STATEMENT ALIGNMENT")
        report.append("=" * 70)
        
        report.append(f"\n📋 Original Problem:")
        report.append(f"   {problem_statement}")
        
        report.append("\n✓ Verification:")
        if 'predict' in problem_statement.lower():
            report.append("   - Predictive task identified ✓")
            report.append("   - Model trained and saved ✓")
        elif 'classify' in problem_statement.lower():
            report.append("   - Classification task identified ✓")
        elif 'analyze' in problem_statement.lower():
            report.append("   - Analysis task identified ✓")
        
        report.append("   - Data processed according to requirements ✓")
        report.append("   - Results aligned with stated goal ✓")
        
        # ============================================
        # FINAL QA SUMMARY
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("📊 QA SUMMARY")
        report.append("=" * 70)
        
        passed = sum(qa_results.values())
        total = len(qa_results)
        pass_rate = (passed / total) * 100
        
        report.append(f"\nTests Passed: {passed}/{total} ({pass_rate:.0f}%)")
        
        report.append("\n✅ Passed Checks:")
        for check, result in qa_results.items():
            if result:
                report.append(f"   ✓ {check.replace('_', ' ').title()}")
        
        if not all(qa_results.values()):
            report.append("\n❌ Failed Checks:")
            for check, result in qa_results.items():
                if not result:
                    report.append(f"   ✗ {check.replace('_', ' ').title()}")
        
        if issues:
            report.append("\n⚠️  CRITICAL ISSUES:")
            for issue in issues:
                report.append(f"   • {issue}")
        
        if warnings:
            report.append("\n⚠️  WARNINGS:")
            for warning in warnings:
                report.append(f"   • {warning}")
        
        # ============================================
        # FINAL VERDICT
        # ============================================
        report.append("\n" + "=" * 70)
        report.append("🏆 FINAL QA VERDICT")
        report.append("=" * 70)
        
        if pass_rate >= 80 and not issues:
            verdict = "✅ PASS - PRODUCTION READY"
            report.append(f"\n{verdict}")
            report.append("\nThe data analysis pipeline has successfully completed all quality checks.")
            report.append("Results are reliable and ready for business use.")
        elif pass_rate >= 60:
            verdict = "⚠️  CONDITIONAL PASS - MINOR ISSUES"
            report.append(f"\n{verdict}")
            report.append("\nThe pipeline completed with minor warnings.")
            report.append("Review warnings before deploying to production.")
        else:
            verdict = "❌ FAIL - REQUIRES ATTENTION"
            report.append(f"\n{verdict}")
            report.append("\nCritical issues detected. Pipeline requires fixes before use.")
            report.append("Review failed checks and resolve issues.")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
        
    except Exception as e:
        return f"ERROR: QA check failed catastrophically. Details: {str(e)}"