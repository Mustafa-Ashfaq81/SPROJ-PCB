"""
LLM Performance Comparison Module
Analyzes LLM performance using consistent methodology with summary CSV files
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class LLMComparisonAnalysis:
    def __init__(self):
        """Initialize LLM comparison analysis"""
        self.summary_csvs_path = Path("..") / "llm" / "plots" / "aggregated_by_model"
        
        # Model configurations
        self.models = {
            'o4-mini': 'o4-mini_summary.csv',
            'gpt-4.1': 'gpt-4.1_summary.csv', 
            'gpt-4.1-mini': 'gpt-4.1-mini_summary.csv',
            'gpt-4.1-nano': 'gpt-4.1-nano_summary.csv'
        }
        
        self.all_models_data = {}
    
    def load_model_summary(self, model_name, csv_filename):
        """Load summary CSV for a specific model"""
        csv_path = self.summary_csvs_path / csv_filename
        
        if not csv_path.exists():
            print(f"Warning: Summary CSV not found for {model_name}: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            
            # Filter out Random Classifier and Top Career Dismissal (non-LLM baselines)
            llm_configs = df[~df['block'].isin(['Random Classifier', 'Top Career Dismissal'])]
            
            # Find best configuration by top-1 accuracy for individual model performance
            best_config_idx = df['top1_acc'].idxmax()
            best_config = df.iloc[best_config_idx]
            
            # Calculate totals across all LLM configurations for overall performance
            total_llm_scenarios = len(llm_configs) * 501  # Each config has 501 scenarios
            total_llm_top1_successes = int(llm_configs['top1_successes'].sum())
            total_llm_top2_successes = int(llm_configs['top2_successes'].sum())
            
            model_data = {
                'model_name': model_name,
                'csv_file': csv_filename,
                # Best individual configuration data
                'best_configuration': best_config['block'],
                'best_total_scenarios': int(best_config['n']),
                'best_top1_successes': int(best_config['top1_successes']),
                'best_top2_successes': int(best_config['top2_successes']),
                'best_top1_accuracy': best_config['top1_acc'] / 100.0,
                'best_top2_accuracy': best_config['top2_acc'] / 100.0,
                'best_top2_improvement': (best_config['top2_acc'] - best_config['top1_acc']) / 100.0,
                # All LLM configurations data for overall performance
                'total_llm_scenarios': total_llm_scenarios,
                'total_llm_top1_successes': total_llm_top1_successes,
                'total_llm_top2_successes': total_llm_top2_successes,
                'num_llm_configurations': len(llm_configs),
                'raw_summary_data': df.to_dict('records')
            }
            
            print(f"{model_name}:")
            print(f"  Best configuration: {best_config['block']}")
            print(f"  Best performance: {best_config['top1_acc']:.1f}% top1, {best_config['top2_acc']:.1f}% top2")
            print(f"  All LLM configs: {len(llm_configs)} configurations, {total_llm_scenarios} total scenarios")
            print(f"  Overall performance: {total_llm_top1_successes/total_llm_scenarios*100:.1f}% top1, {total_llm_top2_successes/total_llm_scenarios*100:.1f}% top2")
            print()
            
            return model_data
            
        except Exception as e:
            print(f"Error loading {model_name} summary: {e}")
            return None
    
    def load_all_model_summaries(self):
        """Load summary data for all LLM models"""
        print("Loading LLM model summaries...")
        print("="*50)
        
        for model_name, csv_filename in self.models.items():
            model_data = self.load_model_summary(model_name, csv_filename)
            if model_data:
                self.all_models_data[model_name] = model_data
        
        print(f"Successfully loaded {len(self.all_models_data)} model summaries")
        return len(self.all_models_data) > 0
    
    def calculate_overall_llm_performance(self):
        """Calculate overall LLM performance across all models and configurations"""
        if not self.all_models_data:
            print("No model data available!")
            return None
        
        total_scenarios = 0
        total_top1_successes = 0
        total_top2_successes = 0
        total_configurations = 0
        
        for model_name, data in self.all_models_data.items():
            total_scenarios += data['total_llm_scenarios']
            total_top1_successes += data['total_llm_top1_successes']
            total_top2_successes += data['total_llm_top2_successes']
            total_configurations += data['num_llm_configurations']
        
        overall_top1_accuracy = total_top1_successes / total_scenarios
        overall_top2_accuracy = total_top2_successes / total_scenarios
        
        overall_performance = {
            'total_scenarios': int(total_scenarios),
            'total_models': len(self.all_models_data),
            'total_configurations': int(total_configurations),
            'total_top1_successes': int(total_top1_successes),
            'total_top2_successes': int(total_top2_successes),
            'top1_accuracy': float(overall_top1_accuracy),
            'top2_accuracy': float(overall_top2_accuracy),
            'top2_improvement': float(overall_top2_accuracy - overall_top1_accuracy),
            'data_source': 'All LLM configurations from summary CSVs (excluding Random Classifier and Top Career Dismissal)'
        }
        
        print("Overall LLM Performance (All Configurations Combined):")
        print("="*50)
        print(f"  Total models: {len(self.all_models_data)}")
        print(f"  Total LLM configurations: {total_configurations}")
        print(f"  Total scenarios: {total_scenarios:,}")
        print(f"  Top-1 accuracy: {overall_top1_accuracy:.1%} ({total_top1_successes:,}/{total_scenarios:,})")
        print(f"  Top-2 accuracy: {overall_top2_accuracy:.1%} ({total_top2_successes:,}/{total_scenarios:,})")
        print(f"  Top-2 improvement: +{overall_top2_accuracy - overall_top1_accuracy:.1%}")
        print(f"  Methodology: All LLM configurations included (corrected from previous best-only approach)")
        
        return overall_performance
    
    def find_best_performing_model(self):
        """Find the best performing individual model"""
        if not self.all_models_data:
            return None, None
        
        best_model_name = None
        best_top1_accuracy = 0
        
        # Find best model by top-1 accuracy
        for model_name, data in self.all_models_data.items():
            if data['best_top1_accuracy'] > best_top1_accuracy:
                best_top1_accuracy = data['best_top1_accuracy']
                best_model_name = model_name
        
        best_model_data = self.all_models_data[best_model_name]
        
        print("Best Performing Individual Model:")
        print("="*40)
        print(f"  Model: {best_model_name}")
        print(f"  Configuration: {best_model_data['best_configuration']}")
        print(f"  Scenarios: {best_model_data['best_total_scenarios']}")
        print(f"  Top-1 accuracy: {best_model_data['best_top1_accuracy']:.1%} ({best_model_data['best_top1_successes']}/{best_model_data['best_total_scenarios']})")
        print(f"  Top-2 accuracy: {best_model_data['best_top2_accuracy']:.1%} ({best_model_data['best_top2_successes']}/{best_model_data['best_total_scenarios']})")
        
        return best_model_name, best_model_data
    
    def compare_with_traditional_models(self, overall_llm_performance, best_model_data, traditional_top1=0.43, traditional_top2=0.58):
        """Compare LLM performance with traditional models"""
        print("\n" + "="*70)
        print("LLM vs TRADITIONAL MODELS COMPARISON")
        print("="*70)
        
        # Overall LLM comparison
        overall_top1_gap = overall_llm_performance['top1_accuracy'] - traditional_top1
        overall_top1_relative = (overall_top1_gap / traditional_top1) * 100
        overall_top2_gap = overall_llm_performance['top2_accuracy'] - traditional_top2
        overall_top2_relative = (overall_top2_gap / traditional_top2) * 100
        
        print(f"\nOverall LLM Performance vs Traditional:")
        print(f"  Top-1: {overall_llm_performance['top1_accuracy']:.1%} vs {traditional_top1:.1%} = +{overall_top1_gap:.1%} ({overall_top1_relative:+.1f}% relative)")
        print(f"  Top-2: {overall_llm_performance['top2_accuracy']:.1%} vs {traditional_top2:.1%} = +{overall_top2_gap:.1%} ({overall_top2_relative:+.1f}% relative)")
        
        # Best model comparison
        best_top1_gap = best_model_data['best_top1_accuracy'] - traditional_top1
        best_top1_relative = (best_top1_gap / traditional_top1) * 100
        best_top2_gap = best_model_data['best_top2_accuracy'] - traditional_top2
        best_top2_relative = (best_top2_gap / traditional_top2) * 100
        
        print(f"\nBest Model ({best_model_data['model_name']}) vs Traditional:")
        print(f"  Top-1: {best_model_data['best_top1_accuracy']:.1%} vs {traditional_top1:.1%} = +{best_top1_gap:.1%} ({best_top1_relative:+.1f}% relative)")
        print(f"  Top-2: {best_model_data['best_top2_accuracy']:.1%} vs {traditional_top2:.1%} = +{best_top2_gap:.1%} ({best_top2_relative:+.1f}% relative)")
        
        # Performance improvement analysis
        traditional_top2_improvement = traditional_top2 - traditional_top1
        best_model_top2_improvement = best_model_data['best_top2_improvement']
        
        print(f"\nTop-2 Improvement Comparison:")
        print(f"  Traditional models: +{traditional_top2_improvement:.1%}")
        print(f"  Best LLM model: +{best_model_top2_improvement:.1%}")
        print(f"  LLM advantage in top-2 capability: +{best_model_top2_improvement - traditional_top2_improvement:.1%}")
        
        comparison_results = {
            'overall_llm_vs_traditional': {
                'top1_gap_absolute': overall_top1_gap,
                'top1_gap_relative_percent': overall_top1_relative,
                'top2_gap_absolute': overall_top2_gap,
                'top2_gap_relative_percent': overall_top2_relative
            },
            'best_model_vs_traditional': {
                'model_name': best_model_data['model_name'],
                'top1_gap_absolute': best_top1_gap,
                'top1_gap_relative_percent': best_top1_relative,
                'top2_gap_absolute': best_top2_gap,
                'top2_gap_relative_percent': best_top2_relative
            },
            'top2_improvement_analysis': {
                'traditional_improvement': traditional_top2_improvement,
                'best_llm_improvement': best_model_top2_improvement,
                'llm_advantage': best_model_top2_improvement - traditional_top2_improvement
            }
        }
        
        return comparison_results
    
    def run_analysis(self):
        """Run the complete LLM comparison analysis"""
        print("LLM Performance Comparison Analysis")
        print("="*50)
        
        # Load all model summaries
        if not self.load_all_model_summaries():
            print("Failed to load model summaries!")
            return None
        
        # Calculate overall performance
        overall_performance = self.calculate_overall_llm_performance()
        if not overall_performance:
            return None
        
        # Find best performing model
        best_model_name, best_model_data = self.find_best_performing_model()
        if not best_model_data:
            return None
        
        # Compare with traditional models
        comparison_results = self.compare_with_traditional_models(overall_performance, best_model_data)
        
        # Generate comprehensive results
        comprehensive_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'methodology': 'Corrected - All LLM configurations included (excluding Random Classifier and Top Career Dismissal baselines)',
            'data_sources': {
                'llm_summaries_path': str(self.summary_csvs_path),
                'models_analyzed': list(self.models.keys()),
                'csv_files_used': list(self.models.values()),
                'total_llm_configurations': sum(data['num_llm_configurations'] for data in self.all_models_data.values()),
                'configurations_per_model': {name: data['num_llm_configurations'] for name, data in self.all_models_data.items()}
            },
            'overall_llm_performance': overall_performance,
            'best_individual_model': {
                'model_name': best_model_name,
                'configuration': best_model_data['best_configuration'],
                'total_scenarios': best_model_data['best_total_scenarios'],
                'top1_accuracy': best_model_data['best_top1_accuracy'],
                'top2_accuracy': best_model_data['best_top2_accuracy'],
                'top1_successes': best_model_data['best_top1_successes'],
                'top2_successes': best_model_data['best_top2_successes'],
                'top2_improvement': best_model_data['best_top2_improvement']
            },
            'all_models_individual_performance': self.all_models_data,
            'comparison_with_traditional': comparison_results,
            'traditional_baseline': {
                'top1_accuracy': 0.43,
                'top2_accuracy': 0.58,
                'data_source': 'traditional_model_results.json'
            }
        }
        
        # Save results
        output_file = Path("processed_data") / "llm_comparison_results.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"\nLLM comparison results saved to: {output_file}")
        
        # Create summary CSV
        summary_data = []
        
        # Add traditional baseline
        summary_data.append({
            'Approach': 'Traditional Models (Best)',
            'Type': 'Baseline',
            'Top-1 Accuracy': '43.0%',
            'Top-2 Accuracy': '58.0%',
            'Scenarios': '500',
            'Configuration': 'Random Forest + Top-2 from probabilities'
        })
        
        # Add overall LLM
        summary_data.append({
            'Approach': 'LLM Overall (All Configurations)',
            'Type': 'Combined',
            'Top-1 Accuracy': f"{overall_performance['top1_accuracy']:.1%}",
            'Top-2 Accuracy': f"{overall_performance['top2_accuracy']:.1%}",
            'Scenarios': str(overall_performance['total_scenarios']),
            'Configuration': f"{overall_performance['total_configurations']} LLM configs from {overall_performance['total_models']} models"
        })
        
        # Add best model
        summary_data.append({
            'Approach': f"LLM Best ({best_model_name})",
            'Type': 'Individual',
            'Top-1 Accuracy': f"{best_model_data['best_top1_accuracy']:.1%}",
            'Top-2 Accuracy': f"{best_model_data['best_top2_accuracy']:.1%}",
            'Scenarios': str(best_model_data['best_total_scenarios']),
            'Configuration': best_model_data['best_configuration']
        })
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = Path("processed_data") / "llm_vs_traditional_summary.csv"
        summary_df.to_csv(csv_file, index=False)
        
        print(f"Summary comparison saved to: {csv_file}")
        
        return comprehensive_results

def main():
    """Main execution function"""
    print("LLM Performance Comparison Module")
    print("="*50)
    
    # Initialize and run analysis
    analyzer = LLMComparisonAnalysis()
    results = analyzer.run_analysis()
    
    if results:
        print("\nLLM comparison analysis completed successfully!")
        
        # Print key findings
        overall = results['overall_llm_performance']
        best = results['best_individual_model']
        
        print("\nKEY FINDINGS:")
        print("="*30)
        print(f"• Overall LLM performance: {overall['top1_accuracy']:.1%} top-1, {overall['top2_accuracy']:.1%} top-2")
        print(f"• Best model ({best['model_name']}): {best['top1_accuracy']:.1%} top-1, {best['top2_accuracy']:.1%} top-2")
        print(f"• Traditional baseline: 43.0% top-1, 58.0% top-2")
        print(f"• Best model advantage: +{(best['top1_accuracy'] - 0.43):.1%} top-1, +{(best['top2_accuracy'] - 0.58):.1%} top-2")
        print(f"• Total LLM configurations analyzed: {results['data_sources']['total_llm_configurations']}")
        print(f"• Total scenarios: {overall['total_scenarios']:,}")
        
    else:
        print("\nLLM comparison analysis failed!")

if __name__ == "__main__":
    main() 