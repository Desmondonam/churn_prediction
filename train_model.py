# ========================================
# 1. DATA PREPROCESSING & MODEL TRAINING
# File: train_model.py
# ========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the churn data"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        
        # Display basic info
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nChurn distribution:\n{df['Churn?'].value_counts()}")
        
        return df
    
    def feature_engineering(self, df):
        """Create new features and encode categorical variables"""
        print("\nPerforming feature engineering...")
        
        # Create a copy
        df_processed = df.copy()
        
        # Drop Phone number (not useful for prediction)
        if 'Phone' in df_processed.columns:
            df_processed = df_processed.drop('Phone', axis=1)
        
        # Encode binary variables
        binary_cols = ['Int\'l Plan', 'VMail Plan']
        for col in binary_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
        
        # Encode target variable
        if 'Churn?' in df_processed.columns:
            df_processed['Churn?'] = df_processed['Churn?'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        
        # One-hot encode State
        if 'State' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['State'], prefix='State')
        
        # One-hot encode Area Code
        if 'Area Code' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['Area Code'], prefix='AreaCode')
        
        # Create interaction features
        if 'Day Mins' in df_processed.columns and 'Day Calls' in df_processed.columns:
            df_processed['Avg_Day_Call_Duration'] = df_processed['Day Mins'] / (df_processed['Day Calls'] + 1)
        
        if 'Eve Mins' in df_processed.columns and 'Eve Calls' in df_processed.columns:
            df_processed['Avg_Eve_Call_Duration'] = df_processed['Eve Mins'] / (df_processed['Eve Calls'] + 1)
        
        if 'Night Mins' in df_processed.columns and 'Night Calls' in df_processed.columns:
            df_processed['Avg_Night_Call_Duration'] = df_processed['Night Mins'] / (df_processed['Night Calls'] + 1)
        
        # Total usage features
        charge_cols = [c for c in df_processed.columns if 'Charge' in c]
        if charge_cols:
            df_processed['Total_Charge'] = df_processed[charge_cols].sum(axis=1)
        
        mins_cols = [c for c in df_processed.columns if 'Mins' in c]
        if mins_cols:
            df_processed['Total_Mins'] = df_processed[mins_cols].sum(axis=1)
        
        calls_cols = [c for c in df_processed.columns if 'Calls' in c and c != 'CustServ Calls']
        if calls_cols:
            df_processed['Total_Calls'] = df_processed[calls_cols].sum(axis=1)
        
        return df_processed
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the churn prediction model"""
        print("\nTraining model...")
        
        # Separate features and target
        X = df.drop('Churn?', axis=1)
        y = df['Churn?']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select the best
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000)
        }
        
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
            print(f"{name} AUC: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
                self.model = model
        
        print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def plot_results(self, y_test, y_pred, y_pred_proba):
        """Plot evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1, 0].set_yticks(range(len(feature_importance)))
            axes[1, 0].set_yticklabels(feature_importance['feature'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].invert_yaxis()
        
        # Prediction Distribution
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='No Churn', color='blue')
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Churn', color='red')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nEvaluation plots saved as 'model_evaluation.png'")
    
    def save_model(self, filepath='churn_model.pkl'):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='churn_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
    
    def predict(self, input_data):
        """Make predictions on new data"""
        # Ensure input_data has all required features
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        probability = self.model.predict_proba(input_scaled)
        
        return prediction, probability


# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load and preprocess data (replace 'churn_data.csv' with your file path)
    df = predictor.load_and_preprocess_data('C:/Users/Admin/Desktop/Freelance/Denis/Projects/Retention_Engine/Data/churn.csv')
    
    # Feature engineering
    df_processed = predictor.feature_engineering(df)
    
    # Train model
    X_test, y_test, y_pred, y_pred_proba = predictor.train(df_processed)
    
    # Plot results
    predictor.plot_results(y_test, y_pred, y_pred_proba)
    
    # Save model
    predictor.save_model('churn_model.pkl')
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)