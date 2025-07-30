from analyzer import ModelAnalyzer
from optimizer import ModelOptimizer
import pandas as pd

class SentimentModelTrainer(ModelAnalyzer, ModelOptimizer):
    """
    Complete asentiment analysis trainer combining all functions

    Inherits from:
    - BaseSentimentTrainer
    - ModelAnalyzer
    - ModelOptimizer

    Uses multiple inheritance to combine all features
    """
    def __init__(self):
        super().__init__()

    def full_pipeline(self, df: pd.DataFrame):
        """
        Complete pipeline from data to optimized model
        """

        self.prepare_data(df)

        self.define_models()

        self.train_and_evaluate()

        comparison_df = self.compare_models()

        best_model_name, best_result = self.analyze_best_model()

        self.plot_model_comparison()

        self.performance_heatmap()

        best_model = self.results[best_model_name]['model']
        self.analyze_features_importance(best_model, best_model_name)

        optimized_model, best_params, optimal_accuracy = self.optimize_best_model(best_model_name)

        if optimized_model:
            self.test_model_on_examples(optimized_model)
            return optimized_model, best_params
        else:
            self.test_model_on_examples(best_model)
            return best_model, None
