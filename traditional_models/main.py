"""
Main Runner Script
Executes the complete cricket dismissal prediction analysis pipeline
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Import our modules
try:
    from data_extraction import CricketDataExtractor
    from traditional_models import TraditionalModelAnalysis
    from llm_comparison import LLMComparisonAnalysis
    from generate_report import EvidenceReportGenerator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all module files are in the same directory.")
    sys.exit(1)

class CricketAnalysisPipeline:
    def __init__(self):
        """Initialize the analysis pipeline"""
        self.start_time = None
        self.results = {}
        
    def print_header(self, title, width=60):
        """Print a formatted header"""
        print("\n" + "="*width)
        print(f" {title.upper()}")
        print("="*width)
    
    def print_step(self, step_num, total_steps, description):
        """Print current step progress"""
        print(f"\n[STEP {step_num}/{total_steps}] {description}")
        print("-" * 50)
    
    def check_prerequisites(self):
        """Check if all required data sources are available"""
        self.print_header("Prerequisites Check")
        
        required_paths = [
            Path("..") / "llm" / "plots" / "aggregated_by_model",
            Path("..") / "cricinfo_data_scrapping" / "aggregated_stats",
            Path("..") / "cricinfo_data_scrapping" / "aggregated_stats_against_teams",
            Path("..") / "cricinfo_data_scrapping" / "aggregated_stats_venue_wise",
            Path("..") / "llm"
        ]
        
        all_good = True
        
        for path in required_paths:
            if path.exists():
                print(f"✓ {path}")
            else:
                print(f"✗ {path} (Missing)")
                all_good = False
        
        # Check for LLM summary CSVs
        llm_summaries = Path("..") / "llm" / "plots" / "aggregated_by_model"
        required_csvs = [
            "o4-mini_summary.csv",
            "gpt-4.1_summary.csv",
            "gpt-4.1-mini_summary.csv",
            "gpt-4.1-nano_summary.csv"
        ]
        
        print("\nLLM Summary CSVs:")
        for csv_file in required_csvs:
            csv_path = llm_summaries / csv_file
            if csv_path.exists():
                print(f"✓ {csv_file}")
            else:
                print(f"✗ {csv_file} (Missing)")
                all_good = False
        
        if not all_good:
            print("\n❌ Prerequisites check failed!")
            print("Please ensure all required data sources are available.")
            return False
        
        print("\n✅ All prerequisites satisfied!")
        return True
    
    def run_data_extraction(self):
        """Step 1: Extract features from cricket data"""
        self.print_step(1, 4, "Data Extraction")
        
        try:
            extractor = CricketDataExtractor()
            features_df, metadata_df = extractor.create_feature_matrix()
            
            if features_df is None or len(features_df) == 0:
                print("❌ Data extraction failed - no features extracted")
                return False
            
            # Save extracted data
            extractor.save_extracted_data(features_df, metadata_df)
            
            self.results['data_extraction'] = {
                'total_scenarios': len(features_df),
                'total_features': features_df.shape[1],
                'status': 'success'
            }
            
            print(f"✅ Data extraction completed successfully!")
            print(f"   • Scenarios extracted: {len(features_df)}")
            print(f"   • Features extracted: {features_df.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Data extraction failed: {e}")
            self.results['data_extraction'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_traditional_models(self):
        """Step 2: Train and evaluate traditional models"""
        self.print_step(2, 4, "Traditional Models Training & Evaluation")
        
        try:
            analyzer = TraditionalModelAnalysis()
            results = analyzer.run_analysis()
            
            if not results:
                print("❌ Traditional models analysis failed")
                return False
            
            # Extract best performance
            valid_results = {name: result for name, result in results.items() if 'error' not in result}
            if not valid_results:
                print("❌ No valid traditional model results")
                return False
            
            best_top1_model = max(valid_results.items(), key=lambda x: x[1]['top1_accuracy'])
            best_top2_model = max(valid_results.items(), key=lambda x: x[1]['top2_accuracy'])
            
            self.results['traditional_models'] = {
                'models_trained': len(valid_results),
                'best_top1_model': best_top1_model[0],
                'best_top1_accuracy': best_top1_model[1]['top1_accuracy'],
                'best_top2_model': best_top2_model[0],
                'best_top2_accuracy': best_top2_model[1]['top2_accuracy'],
                'status': 'success'
            }
            
            print(f"✅ Traditional models analysis completed!")
            print(f"   • Models trained: {len(valid_results)}")
            print(f"   • Best top-1: {best_top1_model[0]} ({best_top1_model[1]['top1_accuracy']:.1%})")
            print(f"   • Best top-2: {best_top2_model[0]} ({best_top2_model[1]['top2_accuracy']:.1%})")
            
            return True
            
        except Exception as e:
            print(f"❌ Traditional models analysis failed: {e}")
            self.results['traditional_models'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_llm_comparison(self):
        """Step 3: Compare LLM performance using consistent methodology"""
        self.print_step(3, 4, "LLM Performance Comparison")
        
        try:
            analyzer = LLMComparisonAnalysis()
            results = analyzer.run_analysis()
            
            if not results:
                print("❌ LLM comparison analysis failed")
                return False
            
            overall_llm = results['overall_llm_performance']
            best_llm = results['best_individual_model']
            
            self.results['llm_comparison'] = {
                'models_analyzed': overall_llm['total_models'],
                'total_scenarios': overall_llm['total_scenarios'],
                'overall_top1_accuracy': overall_llm['top1_accuracy'],
                'overall_top2_accuracy': overall_llm['top2_accuracy'],
                'best_model': best_llm['model_name'],
                'best_top1_accuracy': best_llm['top1_accuracy'],
                'best_top2_accuracy': best_llm['top2_accuracy'],
                'best_configuration': best_llm['configuration'],
                'status': 'success'
            }
            
            print(f"✅ LLM comparison completed!")
            print(f"   • Models analyzed: {overall_llm['total_models']}")
            print(f"   • Overall performance: {overall_llm['top1_accuracy']:.1%} top-1, {overall_llm['top2_accuracy']:.1%} top-2")
            print(f"   • Best model: {best_llm['model_name']} ({best_llm['top1_accuracy']:.1%} top-1, {best_llm['top2_accuracy']:.1%} top-2)")
            
            return True
            
        except Exception as e:
            print(f"❌ LLM comparison failed: {e}")
            self.results['llm_comparison'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_report_generation(self):
        """Step 4: Generate comprehensive evidence report"""
        self.print_step(4, 4, "Evidence Report Generation")
        
        try:
            generator = EvidenceReportGenerator()
            success = generator.run_report_generation()
            
            if not success:
                print("❌ Report generation failed")
                return False
            
            self.results['report_generation'] = {
                'status': 'success',
                'files_generated': [
                    'EVIDENCE_REPORT.md',
                    'evidence_summary.json',
                    'traditional_model_results.json',
                    'llm_comparison_results.json'
                ]
            }
            
            print(f"✅ Evidence report generated successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            self.results['report_generation'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def print_final_summary(self):
        """Print final summary of the entire pipeline"""
        self.print_header("PIPELINE EXECUTION SUMMARY")
        
        execution_time = time.time() - self.start_time
        
        print(f"Total execution time: {execution_time:.1f} seconds")
        print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print step results
        steps = [
            ('Data Extraction', 'data_extraction'),
            ('Traditional Models', 'traditional_models'),
            ('LLM Comparison', 'llm_comparison'),
            ('Report Generation', 'report_generation')
        ]
        
        print(f"\nStep Results:")
        all_successful = True
        
        for step_name, step_key in steps:
            if step_key in self.results:
                status = self.results[step_key]['status']
                if status == 'success':
                    print(f"  ✅ {step_name}: Success")
                else:
                    print(f"  ❌ {step_name}: Failed")
                    all_successful = False
            else:
                print(f"  ⏸️  {step_name}: Not executed")
                all_successful = False
        
        if all_successful:
            self.print_pipeline_results()
        
        return all_successful
    
    def print_pipeline_results(self):
        """Print the key results from the analysis"""
        self.print_header("KEY RESULTS")
        
        # Data overview
        if 'data_extraction' in self.results:
            data_info = self.results['data_extraction']
            print(f"Dataset: {data_info['total_scenarios']} scenarios, {data_info['total_features']} features")
        
        # Traditional models
        if 'traditional_models' in self.results:
            trad_info = self.results['traditional_models']
            print(f"Traditional Best: {trad_info['best_top1_accuracy']:.1%} top-1, {trad_info['best_top2_accuracy']:.1%} top-2")
        
        # LLM models
        if 'llm_comparison' in self.results:
            llm_info = self.results['llm_comparison']
            print(f"LLM Best ({llm_info['best_model']}): {llm_info['best_top1_accuracy']:.1%} top-1, {llm_info['best_top2_accuracy']:.1%} top-2")
            
            # Calculate improvements
            if 'traditional_models' in self.results:
                trad_top1 = self.results['traditional_models']['best_top1_accuracy']
                trad_top2 = self.results['traditional_models']['best_top2_accuracy']
                llm_top1 = llm_info['best_top1_accuracy']
                llm_top2 = llm_info['best_top2_accuracy']
                
                top1_improvement = ((llm_top1 - trad_top1) / trad_top1) * 100
                top2_improvement = ((llm_top2 - trad_top2) / trad_top2) * 100
                
                print(f"LLM Advantage: +{top1_improvement:.1f}% (top-1), +{top2_improvement:.1f}% (top-2) relative improvement")
        
        # Files generated
        print(f"\nGenerated Files:")
        output_dir = Path("processed_data")
        if output_dir.exists():
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    print(f"  • {file_path.name}")
    
    def run_pipeline(self):
        """Run the complete analysis pipeline"""
        self.start_time = time.time()
        
        self.print_header("Cricket Dismissal Prediction Analysis Pipeline")
        print("Quantitative evidence generation for traditional vs LLM model comparison")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Create output directory
        Path("processed_data").mkdir(exist_ok=True)
        
        # Run pipeline steps
        steps = [
            self.run_data_extraction,
            self.run_traditional_models,
            self.run_llm_comparison,
            self.run_report_generation
        ]
        
        for step_func in steps:
            if not step_func():
                print(f"\n❌ Pipeline execution failed at step: {step_func.__name__}")
                self.print_final_summary()
                return False
        
        # Print final summary
        success = self.print_final_summary()
        
        if success:
            print(f"\n🎉 Pipeline execution completed successfully!")
            print(f"📄 Check 'processed_data/EVIDENCE_REPORT.md' for the comprehensive evidence report.")
        
        return success

def main():
    """Main execution function"""
    pipeline = CricketAnalysisPipeline()
    success = pipeline.run_pipeline()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 