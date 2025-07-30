from trainer import BaseSentimentTrainer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class ModelAnalyzer(BaseSentimentTrainer):
    """
    Specialized class for modela analysis and comparison
    Inherits core training from BAseSentimentTrainer
    Focuses on: comparsion, visualization, interpretation
    """
    def compare_models(self) -> pd.DataFrame:
        """Compare all models performance"""
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                # CV Accuracy
                'CV_Accuracy_Mean': results['cv_mean'],
                'CV_Accuracy_STD': results['cv_std'],
                #CV F1
                'CV_F1_Mean': results['cv_f1_mean'],
                'CV_F1_STD': results['cv_f1_std'],
                # Scores
                'Test_Accuracy': results['test_accuracy'],
                'Test_F1': results['test_f1'],
                # Time
                'Training_Time': results['train_time'],
                'Inference_Time': results['inference_time'],
                'Time_Taken': results['time_taken']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('CV_F1_Mean', ascending=False)

        print("\n" + "=="*60)
        print("\t\t\tMODEL COMPARISON RESULTS")
        print("=="*60)
        print(comparison_df.to_string(index=False))

        return comparison_df
    
    def analyze_best_model(self):
        """Analysis of the best performing model"""
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_accuracy'])
        best_model_result = self.results[best_model_name]

        print(f"\n" + "="*60)
        print(f"Analysis of the best model: {best_model_name}")
        print(f"="*60)

        cm = confusion_matrix(self.y_test, best_model_result['prediction'])
        
        labels = ['negative', 'neutral', 'positive']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='crest', 
                    xticklabels= labels,
                    yticklabels= labels)
        plt.title(f"Confusion Matrix - {best_model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return best_model_name, best_model_result
    
    def plot_model_comparison(self) -> None:
        """Visualize model comparison"""
        models = list(self.results.keys())

        metrics = ['test_accuracy', 'cv_mean', 'test_f1', 'cv_std']
        metrics_plots = {metric: [self.results[model][metric] for model in models] for metric in metrics}

        cv_means = metrics_plots['cv_mean']
        cv_stds = metrics_plots['cv_std']
        accuracies = metrics_plots['test_accuracy']
        f1_scores = metrics_plots['test_f1']

        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))

        bar1 = ax.bar(x - width, cv_means, width, yerr=cv_stds, label='CV Accuracy', alpha=0.8, capsize=5)
        bar2 = ax.bar(x, accuracies, width, label='Test Accuracy', alpha=0.8)
        bar3 = ax.bar(x + width, f1_scores, label='Test F1', alpha=0.8)

        ax.set_xlabel('Models')
        ax.set_ylabel('Metrics')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bars in [bar1, bar2, bar3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height+0.01, 
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

    def performance_heatmap(self) -> None:
        metrics = ['CV Accuracy', 'Test Accuracy', 'Test F1', 'Training Time']

        data = []
        for model in self.models:
            row = [
                self.results[model]['cv_mean'],
                self.results[model]['test_accuracy'],
                self.results[model]['test_f1'],
                1 / (self.results[model]['train_time'] + 0.001), 
            ]
            data.append(row)

        df_heatmap = pd.DataFrame(data, index=self.models, columns=metrics)

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_heatmap, annot=True, cmap='RdYlGn', center=0.7,
                    fmt='.3f', cbar_kws={'label': 'Performance  Score'})
        plt.title('Model Performance Heatmap')
        plt.tight_layout()
        plt.show()

    def analyze_features_importance(self, model, model_name) -> None:
        """Analyize what the model learned"""
        print(f"\nFeature Importance Analysis for {model_name}")
        print("==="*20)
        
        try:
            features_names = model.named_steps['tfidf'].get_feature_names_out()
        except:
            print(f"Cannot get features names from {model_name}")
            return
        
        model_steps_classifier = model.named_steps['classifier']
        top_features = []

        if hasattr(model_steps_classifier, 'feature_importances_'):
            importances = model_steps_classifier.feature_importances_
            top_indices = importances.argsort()[-20:][::-1]
            top_features = [(features_names[i], importances[i]) for i in top_indices]

        elif hasattr(model_steps_classifier, 'coef_'):
            coef = model_steps_classifier.coef_
            coef_mean = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef[0])
            top_indices = coef_mean.argsort()[-20:][::-1]
            top_features = [(features_names[i], coef_mean[i]) for i in top_indices]

        else:
            print(f"The model {model_name} does not support feature importance")

        print("Top 20 Important Features:")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
