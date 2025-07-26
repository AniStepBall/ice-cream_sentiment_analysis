import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

class SentimentModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def preparing_data(self, df, text_column='clean_text', target_column='sentiment_category'):
        """Prepare data for model training"""
        print("Preparing data for training...")
        
        X = df[text_column].fillna('')
        y = df[target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Class distribution in training")
        print(self.y_train.value_counts())

    def define_the_models(self):
        """Define all baseline models"""
        tfidVector = TfidfVectorizer(max_features=10000, stop_words='english')
        self.models = {
            'Logistic Regression':LogisticRegression(class_weight='balanced', random_state=42),
            'Navie Bayes': MultinomialNB(),
            'Linear Support Vector': LinearSVC(class_weight='balanced', max_iter=1000, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42),
            'Voting Hard': VotingClassifier(estimators=[
                ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
                ('nb', MultinomialNB()),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)),
            ], voting='hard')
        }

        for name, classifier in self.models.items():
            self.models[name] = Pipeline([
                ('tfidf', tfidVector),
                ('classifier', classifier)
            ])
        
        print(f"Defined {len(self.models)} baseline models")

    def train_and_evaluate_models(self):
        """Training the models and perfoming cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")

            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            
            start_fit_time = time.time()
            model.fit(self.X_train, self.y_train)
            end_fit_time = time.time()

            start_pred_time = time.time()
            y_pred = model.predict(self.X_test)
            end_pred_time = time.time()

            self.results[name] = {
                'model': model,
                'cv_score': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy_score(self.y_test, y_pred),
                'test_f1': f1_score(self.y_test, y_pred, average='weighted'),
                'prediction': y_pred,
                'classification_report': classification_report(self.y_test, y_pred),
                'train_time': round((end_fit_time - start_fit_time), 4),
                'inference_time': round((end_pred_time - start_pred_time), 4),
                'time_taken': round((end_pred_time - start_fit_time), 5),
            }

            print(f"CV Accuracy: {cv_scores.mean():.5f} (+/- {cv_scores.std()*2:.5f})")
            print(f"Accuracy Score: {accuracy_score(self.y_test, y_pred)}")
            print(f"F1 Score (weighted average): {f1_score(self.y_test, y_pred, average='weighted')}")

    def comparing_models(self):
        """Compare all models performace"""
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                # CV 
                'CV_Mean': results['cv_mean'],
                'CV_STD': results['cv_std'],
                # Scores
                'Test_Accuracy': results['test_accuracy'],
                'Test_F1': results['test_f1'],
                # Time
                'Training_Time': results['train_time'],
                'Inference_Time': results['inference_time'],
                'Time_Taken': results['time_taken']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_F1', ascending=False)

        print("\n" + "=="*60)
        print("\t\t\tMODEL COMPARISON RESULTS")
        print("=="*60)
        print(comparison_df.to_string(index=False))

        return comparison_df
    
    def analyze_best_models(self):
        """Analysis of the best perfoming model"""
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_f1'])  #inital test_accuarcy
        best_model_result = self.results[best_model_name]

        print(f"\n" + "="*60)
        print(f"Analysis of the model: {best_model_name}")
        print(f"="*60)

        cm = confusion_matrix(self.y_test, best_model_result['prediction'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='crest', 
                    xticklabels=['negative', 'neutral', 'positive'],
                    yticklabels=['negative', 'neutral', 'positive'])
        plt.title(f"Confusion Matrix - {best_model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return best_model_name, best_model_result
    
    def plot_model_comparison(self):
        """Visualize model comparison"""
        models = list(self.results.keys())

        metrics = ['test_accuracy', 'cv_mean', 'test_f1']
        metrics_plots = {metric: [self.results[model][metric] for model in models] for metric in metrics}

        accuracies = metrics_plots['test_accuracy']
        cv_means = metrics_plots['cv_mean']
        f1_scores = metrics_plots['test_f1']

        x = np.arange(len(models))
        
        fig, axes = plt.subplots(figsize=(10, 6))
        #0.35 = width;
        axes.plot(x, cv_means, marker='o', label='CV Mean', alpha=0.8)
        axes.plot(x, accuracies, marker='o', label='Accuracy', alpha=0.8)
        axes.plot(x, f1_scores, marker='o', label='F1 Score', alpha=0.8)

        axes.set_xlabel('Models')
        axes.set_ylabel('Metrics')
        axes.set_title('Model Performace Comparison')
        axes.set_xticks(x)
        axes.set_xticklabels(models, rotation=45)
        axes.legend()

        plt.tight_layout()
        plt.show()

    def optimize_best_model(self, best_model_name):
        """Optimize the best performing model"""
        print(f"Optimizing {best_model_name}")

        model = self.models[best_model_name]

        param_grids = {
            'Logistic Regression':{
                'classifier__max_iter': [10, 25, 50, 100],
                'classifier__penalty': ['l2'],
            },
            'Navie Bayes':{
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
            },
            'Linear Support Vector':{
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__max_iter': [100, 250, 500, 1000],
            },
            'Voting Softly':{
                'classifier__voting': ['hard', 'soft'],
                'classifier__weight': [None, [1,2,1]],
            },
            'Gradient Boosting':{
                'classifier__max_depth': [5, 10, 15],
                'classifier__n_estimators': [50, 100, 150],
            }
        }

        param_grid = param_grids.get(best_model_name, {})

        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=5, n_jobs = -1, verbose=2, scoring='accuracy'
            )

            grid_search.fit(self.X_train, self.y_train)

            best_optimize_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            y_pred_optimized = best_optimize_model.predict(self.X_test)
            optimial_accuracy = accuracy_score(self.y_test, y_pred_optimized)

            print(f"Best parameters: {best_params}")
            print(f"Optimized accuracy {optimial_accuracy:.3f}")
            print(f"Improvement: {optimial_accuracy - self.results[best_model_name]['test_accuracy']:.5f}")

            return best_optimize_model, best_params, optimial_accuracy
        else:
            print(f"No parameters grid defined for {best_model_name}")
            return None, None, None

    def analyze_features_importance(self, model, model_name):
        """Analyize what the model learned"""
        print(f"\nFeature Importance Analysis for {model_name}")
        print("==="*20)
        
        try:
            features_names = model.named_steps['tfidf'].get_feature_names_out()
        except:
            print(f"Can get features names from {model_name}")
            return
        
        model_steps_classifer = model.named_steps['classifier']
        top_features = []

        if hasattr(model_steps_classifer, 'feature_importances_'):
            importances = model_steps_classifer.feature_importances_
            top_indices = importances.argsort()[-20:][::-1]
            top_features = [(features_names[i], importances[i]) for i in top_indices]

        elif hasattr(model_steps_classifer, 'coef_'):
            coef = model_steps_classifer.coef_
            coef_mean = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef[0])
            top_indices = coef_mean.argsort()[-20:][::-1]
            top_features = [(features_names[i], coef_mean[i]) for i in top_indices]

        else:
            print(f"The model {model_name} does not support feature importance")

        print("Top 20 Important Features:")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")

    def test_model_on_examples(self, model):
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
