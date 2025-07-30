import time
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

MAX_FEATURES = 5000
RANDOM = 42

class BaseSentimentTrainer:
    """
    Base class handling core training functionality
    Focuses on: data preparation, model defintion, basic training
    """
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, df:pd.DataFrame, text_column='clean_text', target_column='sentiment_category') -> None:
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

    def define_models(self) -> None:
        """Define all baseline models"""
        #consider tfidf_vectorizer for each
        tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
        self.models = {
            'Logistic Regression':LogisticRegression(class_weight='balanced', random_state=RANDOM),
            'Naive Bayes': MultinomialNB(),
            'Linear Support Vector': LinearSVC(class_weight='balanced', max_iter=1000, random_state=RANDOM),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=RANDOM),
            'Voting Hard': VotingClassifier(estimators=[
                ('lr', LogisticRegression(class_weight='balanced', random_state=RANDOM)),
                ('nb', MultinomialNB()),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM)),
            ], voting='hard')
        }

        for name, classifier in self.models.items():
            self.models[name] = Pipeline([
                ('tfidf', tfidf_vectorizer),
                ('classifier', classifier)
            ])
        
        print(f"Defined {len(self.models)} baseline models")

    def train_and_evaluate(self) -> None:
        """Training the models and performing cross-validation"""
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            #'balanced_accuracy': 'balanced_accuracy' 
        }

        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")

            cv_results = cross_validate(model, self.X_train, self.y_train,
                                        cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
           
            cv_accuracy_scores = cv_results['test_accuracy']
            cv_f1_scores = cv_results['test_f1_weighted']

            start_fit_time = time.time()
            model.fit(self.X_train, self.y_train)
            end_fit_time = time.time()

            start_pred_time = time.time()
            y_pred = model.predict(self.X_test)
            end_pred_time = time.time()

            self.results[name] = {
                'model': model,

                'cv_score': cv_accuracy_scores,
                'cv_mean': cv_accuracy_scores.mean(),
                'cv_std': cv_accuracy_scores.std(),

                'cv_f1_score': cv_f1_scores,
                'cv_f1_mean': cv_f1_scores.mean(),
                'cv_f1_std': cv_f1_scores.std(),

                'test_accuracy': accuracy_score(self.y_test, y_pred),
                'test_f1': f1_score(self.y_test, y_pred, average='weighted'),
                
                'prediction': y_pred,
                'classification_report': classification_report(self.y_test, y_pred),
                
                'train_time': round((end_fit_time - start_fit_time), 4),
                'inference_time': round((end_pred_time - start_pred_time), 4),
                'time_taken': round((end_pred_time - start_fit_time), 5),
            }

        print(f"CV Accuracy: {cv_accuracy_scores.mean():.5f} (+/- {cv_accuracy_scores.std()*2:.5f})")
        print(f"CV F1 Weighted: {cv_f1_scores.mean():.5f} (+/- {cv_f1_scores.std()*2:.5f})")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.5f}")
        print(f"Test F1 Weighted: {f1_score(self.y_test, y_pred, average='weighted'):.5f}")
