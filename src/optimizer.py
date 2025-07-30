from trainer import BaseSentimentTrainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class ModelOptimizer(BaseSentimentTrainer):
    """
    Specialized class for model optimization
    Inherits core training from BaseSentimentTrainer
    Focuses on: hyperparameter tuning, optimization, testing
    """
    def optimize_best_model(self, best_model_name):
        """Optimize the best performing model"""
        print(f"Optimizing {best_model_name}")

        model = self.models[best_model_name]

        param_grids = {
            'Logistic Regression':{
                'classifier__max_iter': [10, 25, 50, 100],
                'classifier__penalty': ['l2'],
            },
            'Naive Bayes':{
                'classifier__alpha': [0.2, 0.5, 2.0, 20],
                'classifier__fit_prior': [True, False],
            },
            'Linear Support Vector':{
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__max_iter': [100, 250, 500, 1000],
            },
            'Voting Hard':{
                'classifier__voting': ['hard', 'soft'],
                'classifier__weights': [None, [1,2,1]],
            },
            'Gradient Boosting':{
                'classifier__max_depth': [5, 10, 15],
                'classifier__n_estimators': [50, 100, 150],
            }
        }

        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted'
        }

        param_grid = param_grids.get(best_model_name, {})

        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=5, n_jobs = -1, verbose=2, scoring=scoring, refit='f1_weighted'
            )

            grid_search.fit(self.X_train, self.y_train)

            best_optimize_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            y_pred_optimized = best_optimize_model.predict(self.X_test)
            optimial_accuracy = accuracy_score(self.y_test, y_pred_optimized)

            print(f"Best parameters: {best_params}")
            print(f"Best F1 Score: {best_score}")
            print(f"Optimized accuracy: {optimial_accuracy:.3f}")
            print(f"Improvement: {optimial_accuracy - self.results[best_model_name]['test_accuracy']:.5f}")

            return best_optimize_model, best_params, optimial_accuracy
        else:
            print(f"No parameters grid defined for {best_model_name}")
            return None, None, None

    def test_model_on_examples(self, model) -> None:
        """Test model on hand-picked examples"""
        test_examples = [
            "Super good, don't get me wrong. But I came for the caramel and brownies, not the sweet cream. "
            "The taste of this was amazing, but the ratio of brownie to sweet cream was disappointing. "
            "Liked it regardless but probably won't buy again simply because it didn't live up to its promising package. "
            "I'll find another one that has a better ratio and wayyy more yummy chewy brownies. "
            "Overall, good flavor, texture, idea, and brownies."  
        ]

        print("\nTesting model on example reviews:")
        print("=="*25)

        for example in test_examples:
            prediction = model.predict([example])[0]
            try:
                probability = model.predict_proba([example])[0]
                max_prob = max(probability)
                print(f"Review: {example}")
                print(f"Prediction: {prediction} (confidence: {max_prob:.3f})")
                print("---"*10)
            except:
                print(f"Review: {example}")
                print(f"Prediction: {prediction}")
                print("---"*10)
