import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

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

    def define_the_models(self):# consider merge this function and the next 2 [define, train, evaluate, compare model]
        """Define all baseline models"""
        self.models = {     #think whether to adjust these or not [i.e., add or remove models, change the pipeline etc]
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'Navie Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ]),
            'Random Forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'SVM': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', SVC(kernel='rbf', random_state=42))
            ])
        }

        print(f"Defined {len(self.models)} baseline models")

    def train_and_evaluate_models(self):
        """Training the models and perfoming cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")
#start
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')

            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)
#end
            self.results[name] = {      #adding timer
                'model': model,
                'cv_score': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy_score(self.y_test, y_pred),
                'predication': y_pred,  #spelling
                'classification_report': classification_report(self.y_test, y_pred)
            }   #'timer': end - start

            print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            print(f"AS Accuracy: {accuracy_score(self.y_test, y_pred)}")
            #Consider precision, recall and f1 score

    def comparing_models(self): #consider adding time took
        """Compare all models performace"""
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'CV_Mean': results['cv_mean'],
                'CV_STD': results['cv_std'],
                'Test_Accuracy': results['test_accuracy']
            })  #'Time_Taken': results['timer']

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)

        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison_df.to_string(index=False))

        return comparison_df
    
    def analyze_best_models(self):
        """Analysis of the best perfoming model"""
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_accuracy'])
        best_model_result = self.results[best_model_name]

        print(f"\n" + "="*60)
        print(f"Analysis of the model: {best_model_name}")      #Consider maybe all the models instead?
        print(f"="*60)

        cm = confusion_matrix(self.y_test, best_model_result['predication'])    #spelling
        
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
        accuracies = [self.results[model]['test_accuracy'] for model in models]
        cv_means = [self.results[model]['cv_mean'] for model in models]

        x = np.arange(len(models))
        
        fig, axes = plt.subplots(figsize=(10, 6))
        #0.35 = width; consider different graph types?
        axes.bar(x - 0.35/2, cv_means, 0.35, label='CV Mean', alpha=0.8)
        axes.bar(x + 0.35/2, accuracies, 0.35, label='Test Accuracy', alpha=0.8)

        axes.set_xlabel('Models')
        axes.set_ylabel('Accuracy')
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

        param_grids ={
            'Logistic Regression':{
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2'],
                'tfidf__max_features': [3000, 5000, 8000]
            },
            'Navie Bayes':{
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
                'tfidf__max_features': [3000, 5000, 8000]
            },
            'Random Forest':{
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0],
                'tfidf__max_features': [3000, 5000, 8000]
            },
            'SVM':{
                'classifier__C': [0.1, 1, 10],
                'classifier__gamma': ['scale', 'auto'],
                'tfidf__max_features': [3000, 5000]
            }
        }

        param_grid = param_grids.get(best_model_name, {})   #this variable sucks, change it later

        if param_grid:
            grid_search = GridSearchCV(     #Checks past for better
                model, param_grid, cv=3, scoring='accuracy',
                n_jobs=1, verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)

            best_optimize_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            y_pred_optimized = best_optimize_model.predict(self.X_test)
            optimial_accuracy = accuracy_score(self.y_test, y_pred_optimized)

            print(f"Best parameters: {best_params}")
            print(f"Optimized accuracy {optimial_accuracy:.3f}")
            print(f"Improvement: {optimial_accuracy - self.results[best_model_name]['test_accuracy']:.3f}")

            return best_optimize_model, best_params, optimial_accuracy
        else:
            print(f"No parameters grid defined for {best_model_name}")
            return None, None, None

    def analyze_features_importance(self, model, model_name):#i dont know man, lets remove this function
        """Analyize what the model learned"""
        print(f"\nFeature Importance Analysis for {model_name}")
        print("==="*20)

        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            features_names = model.named_steps['tfidf'].get_feature_names_out()
            importances = model.named_steps['classifier'].features_importances_

            top_indices = importances.argsort()[-20:][::-1]
            top_features = [(features_names[i], importances[i]) for i in top_indices]

            print("Top 20 Important Features:")
            for feature, importance in top_features:
                print(f"{feature}: {importance:.4f}")

        elif hasattr(model.named_steps['classifier'], 'coef_'):
            features_names = model.named_steps['tfidf'].get_feature_names_out()

            if len(model.named_steps['classifier'].coef_.shape) > 1:
                coef_mean = np.mean(np.abs(model.named_steps['classifier'].coef_), axis=0)
            else:
                coef_mean = np.abs(model.named_steps['classifier'].coef_[0])

            top_indices = coef_mean.argsort()[-20:][::-1]
            top_features = [(features_names[i], coef_mean[i]) for i in top_indices]

            print("Top 20 Important Features:")
            for feature, importance in top_features:
                print(f"{feature}: {importance:.4f}")

    def test_model_on_examples(self, model):
        """Test model on hand-picked examples"""
        test_examples = [
            "Super good, don't get me wrong. But I came for the caramel and brownies, not the sweet cream."
            "The taste of this was amazing, but the ratio of brownie to sweet cream was disappointing."
            "Liked it regardless but probably won't buy again simply because it didn't live up to its promising package." 
            "I'll find another one that has a better ratio and wayyy more yummy chewy brownies."
            "Overall, good flavor, texture, idea, and brownies." 
            "Not so great caramel/sweet cream/ brownie RATIO."
            "Just add more brownies. Please."
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
