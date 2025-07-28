"""
Traditional Models Training and Evaluation Module
Trains multiple traditional ML models and evaluates top-1 and top-2 accuracy
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import Counter
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

class TraditionalModelAnalysis:
    def __init__(self):
        """Initialize the traditional model analysis"""
        self.models = {}
        self.results = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Define models to train
        self.model_configs = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=5
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                multi_class='ovr'
            ),
            'SVM': SVC(
                random_state=42, 
                probability=True,  # Enable probability for top-2 accuracy
                kernel='rbf'
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['XGBoost'] = XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
    
    def load_data(self):
        """Load the extracted features and metadata"""
        try:
            features_df = pd.read_csv('processed_data/features.csv')
            metadata_df = pd.read_csv('processed_data/metadata.csv')
            
            print(f"Loaded data:")
            print(f"  Features shape: {features_df.shape}")
            print(f"  Metadata shape: {metadata_df.shape}")
            
            # Remove unknown ground truth entries
            valid_indices = metadata_df['ground_truth'] != 'Unknown'
            features_df = features_df[valid_indices]
            metadata_df = metadata_df[valid_indices]
            
            print(f"After filtering unknown ground truth:")
            print(f"  Features shape: {features_df.shape}")
            print(f"  Ground truth distribution:")
            print(metadata_df['ground_truth'].value_counts())
            
            return features_df, metadata_df
            
        except FileNotFoundError as e:
            print(f"Error: Could not load data files. {e}")
            print("Please run data_extraction.py first.")
            return None, None
    
    def get_baseline_top2_accuracy(self, y_test):
        """Calculate baseline top-2 accuracy using most frequent classes"""
        class_counts = Counter(y_test)
        most_common_classes = [cls for cls, _ in class_counts.most_common(2)]
        
        # For each test sample, check if true class is in top-2 most frequent
        correct_top2 = sum(1 for true_class in y_test if true_class in most_common_classes)
        baseline_top2_accuracy = correct_top2 / len(y_test)
        
        return baseline_top2_accuracy, most_common_classes
    
    def calculate_top2_accuracy_from_probabilities(self, model, X_test, y_test, model_name):
        """Calculate top-2 accuracy using model's probability predictions"""
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
                top2_correct = 0
                
                for i, probs in enumerate(probabilities):
                    # Get indices of top-2 highest probabilities
                    top2_indices = np.argsort(probs)[-2:]
                    top2_classes = [self.label_encoder.classes_[idx] for idx in top2_indices]
                    actual_class = self.label_encoder.classes_[y_test[i]]
                    
                    if actual_class in top2_classes:
                        top2_correct += 1
                
                top2_accuracy = top2_correct / len(y_test)
                return top2_accuracy, top2_correct, len(y_test)
            else:
                print(f"Warning: {model_name} does not support probability prediction")
                return None, None, None
                
        except Exception as e:
            print(f"Error calculating top-2 accuracy for {model_name}: {e}")
            return None, None, None
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate both top-1 and top-2 accuracy"""
        results = {}
        
        print("\nTraining and evaluating models...")
        print("="*60)
        
        # Calculate baseline top-2 accuracy
        baseline_top2, most_common = self.get_baseline_top2_accuracy(y_test)
        print(f"Baseline top-2 accuracy (most frequent classes): {baseline_top2:.1%}")
        print(f"Most frequent classes: {most_common}")
        print()
        
        for model_name, model in self.model_configs.items():
            print(f"Training {model_name}...")
            
            try:
                # Train model
                if model_name == 'Naive Bayes':
                    # Naive Bayes requires non-negative features
                    X_train_nb = X_train - X_train.min() + 1
                    X_test_nb = X_test - X_test.min() + 1
                    model.fit(X_train_nb, y_train)
                    y_pred = model.predict(X_test_nb)
                    
                    # Calculate top-2 accuracy for Naive Bayes
                    top2_acc, top2_correct, total_test = self.calculate_top2_accuracy_from_probabilities(
                        model, X_test_nb, y_test, model_name
                    )
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate top-2 accuracy
                    top2_acc, top2_correct, total_test = self.calculate_top2_accuracy_from_probabilities(
                        model, X_test, y_test, model_name
                    )
                
                # Calculate top-1 accuracy
                top1_acc = accuracy_score(y_test, y_pred)
                
                # Calculate other metrics
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'top1_accuracy': top1_acc,
                    'top2_accuracy': top2_acc if top2_acc is not None else baseline_top2,
                    'top2_correct': top2_correct if top2_correct is not None else int(baseline_top2 * len(y_test)),
                    'total_test_samples': len(y_test),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'prediction_distribution': {str(k): int(v) for k, v in Counter(y_pred).items()}
                }
                
                # Print results
                print(f"  Top-1 Accuracy: {top1_acc:.1%}")
                print(f"  Top-2 Accuracy: {top2_acc:.1%}" if top2_acc is not None else f"  Top-2 Accuracy: {baseline_top2:.1%} (baseline)")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                
                # Check prediction diversity
                unique_predictions = len(set(y_pred))
                total_classes = len(self.label_encoder.classes_)
                print(f"  Prediction diversity: {unique_predictions}/{total_classes} classes")
                print(f"  Most frequent prediction: {Counter(y_pred).most_common(1)[0]}")
                print()
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {
                    'error': str(e),
                    'top1_accuracy': 0,
                    'top2_accuracy': 0
                }
        
        return results
    
    def analyze_class_imbalance(self, y):
        """Analyze class distribution and imbalance"""
        class_counts = Counter(y)
        total_samples = len(y)
        
        print("Class Distribution Analysis:")
        print("="*40)
        
        for class_name, count in class_counts.most_common():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        most_common_count = class_counts.most_common(1)[0][1]
        least_common_count = class_counts.most_common()[-1][1]
        imbalance_ratio = most_common_count / least_common_count
        
        print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"Majority class: {class_counts.most_common(1)[0][0]} ({class_counts.most_common(1)[0][1]} samples)")
        
        return class_counts, imbalance_ratio
    
    def run_analysis(self):
        """Run the complete traditional model analysis"""
        print("Traditional Models Analysis")
        print("="*50)
        
        # Load data
        features_df, metadata_df = self.load_data()
        if features_df is None:
            return None
        
        # Prepare features and target
        X = features_df.values
        y = metadata_df['ground_truth'].values
        
        print(f"\nDataset Information:")
        print(f"  Total samples: {len(X)}")
        print(f"  Total features: {X.shape[1]}")
        print(f"  Unique classes: {len(np.unique(y))}")
        
        # Analyze class imbalance
        class_counts, imbalance_ratio = self.analyze_class_imbalance(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with stratification to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, 
            test_size=0.2, 
            stratify=y_encoded, 
            random_state=42
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        
        # Train and evaluate models
        self.results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Generate summary
        self.generate_summary(class_counts, imbalance_ratio)
        
        return self.results
    
    def generate_summary(self, class_counts, imbalance_ratio):
        """Generate and save analysis summary"""
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        
        # Find best performing models
        valid_results = {name: result for name, result in self.results.items() if 'error' not in result}
        
        if not valid_results:
            print("No valid results to summarize!")
            return
        
        # Sort by top-1 accuracy
        sorted_by_top1 = sorted(valid_results.items(), key=lambda x: x[1]['top1_accuracy'], reverse=True)
        
        print("\nTop-1 Accuracy Ranking:")
        for i, (model_name, result) in enumerate(sorted_by_top1, 1):
            print(f"  {i}. {model_name}: {result['top1_accuracy']:.1%}")
        
        # Sort by top-2 accuracy
        sorted_by_top2 = sorted(valid_results.items(), key=lambda x: x[1]['top2_accuracy'], reverse=True)
        
        print("\nTop-2 Accuracy Ranking:")
        for i, (model_name, result) in enumerate(sorted_by_top2, 1):
            print(f"  {i}. {model_name}: {result['top2_accuracy']:.1%}")
        
        # Best overall performance
        best_top1_model = sorted_by_top1[0]
        best_top2_model = sorted_by_top2[0]
        
        print(f"\nBest Top-1 Performance: {best_top1_model[0]} ({best_top1_model[1]['top1_accuracy']:.1%})")
        print(f"Best Top-2 Performance: {best_top2_model[0]} ({best_top2_model[1]['top2_accuracy']:.1%})")
        
        # Save detailed results
        summary_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': sum(self.results[list(self.results.keys())[0]]['total_test_samples'] for _ in [1]) * 5,  # Approximate total
                'test_samples': list(valid_results.values())[0]['total_test_samples'],
                'total_features': len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else "Unknown",
                'unique_classes': len(self.label_encoder.classes_),
                'class_distribution': {str(k): int(v) for k, v in dict(class_counts).items()},
                'imbalance_ratio': imbalance_ratio
            },
            'model_results': {},
            'best_performance': {
                'top1_model': best_top1_model[0],
                'top1_accuracy': best_top1_model[1]['top1_accuracy'],
                'top2_model': best_top2_model[0], 
                'top2_accuracy': best_top2_model[1]['top2_accuracy']
            }
        }
        
        # Add individual model results
        for model_name, result in valid_results.items():
            if 'error' not in result:
                summary_data['model_results'][model_name] = {
                    'top1_accuracy': result['top1_accuracy'],
                    'top2_accuracy': result['top2_accuracy'],
                    'top2_correct': result['top2_correct'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'prediction_distribution': result['prediction_distribution'],
                    'unique_predictions': len(result['prediction_distribution'])
                }
        
        # Save summary
        output_file = Path("processed_data") / "traditional_model_results.json"
        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Also save a CSV summary for easy viewing
        csv_data = []
        for model_name, result in valid_results.items():
            csv_data.append({
                'Model': model_name,
                'Top-1 Accuracy': f"{result['top1_accuracy']:.1%}",
                'Top-2 Accuracy': f"{result['top2_accuracy']:.1%}",
                'Precision': f"{result['precision']:.3f}",
                'Recall': f"{result['recall']:.3f}",
                'F1-Score': f"{result['f1_score']:.3f}",
                'Unique Predictions': len(result['prediction_distribution'])
            })
        
        csv_summary = pd.DataFrame(csv_data)
        csv_file = Path("processed_data") / "traditional_model_summary.csv"
        csv_summary.to_csv(csv_file, index=False)
        
        print(f"Summary table saved to: {csv_file}")
        
        return summary_data

def main():
    """Main execution function"""
    print("Traditional Models Training and Evaluation")
    print("="*50)
    
    # Initialize and run analysis
    analyzer = TraditionalModelAnalysis()
    results = analyzer.run_analysis()
    
    if results:
        print("\nTraditional model analysis completed successfully!")
    else:
        print("\nTraditional model analysis failed!")

if __name__ == "__main__":
    main() 