"""
Churn Prediction - Model Training & Evaluation
Trains Gradient Boosting models with cross-validation and evaluation
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, auc, precision_recall_curve, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.feature_cols = None
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """Prepare data for modeling"""
        print("=" * 60)
        print("STEP 1: DATA PREPARATION")
        print("=" * 60)
        
        # Separate features and target
        self.feature_cols = [col for col in self.df.columns if col != 'Churn']
        self.X = self.df[self.feature_cols]
        self.y = self.df['Churn']
        
        print(f"Features: {self.X.shape[1]}")
        print(f"Samples: {self.X.shape[0]}")
        print(f"Churn Rate: {self.y.mean():.2%}")
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Train churn rate: {self.y_train.mean():.2%}")
        print(f"Test churn rate: {self.y_test.mean():.2%}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost(self):
        """Train Gradient Boosting model (Model 1)"""
        print("\n" + "=" * 60)
        print("STEP 2: TRAINING GRADIENT BOOSTING MODEL (Model 1)")
        print("=" * 60)
        
        xgb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            validation_fraction=0.2,
            n_iter_no_change=20
        )
        
        # Train
        xgb_model.fit(self.X_train, self.y_train)
        
        self.models['xgboost'] = xgb_model
        
        # Predictions
        y_pred_xgb = xgb_model.predict(self.X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluation
        auc_xgb = roc_auc_score(self.y_test, y_pred_proba_xgb)
        f1_xgb = f1_score(self.y_test, y_pred_xgb)
        
        print(f"Gradient Boosting ROC-AUC: {auc_xgb:.4f}")
        print(f"Gradient Boosting F1-Score: {f1_xgb:.4f}")
        print("\nClassification Report (Gradient Boosting):")
        print(classification_report(self.y_test, y_pred_xgb, target_names=['No Churn', 'Churn']))
        
        return xgb_model, y_pred_xgb, y_pred_proba_xgb
    
    def train_lightgbm(self):
        """Train Random Forest model (Model 2)"""
        print("\n" + "=" * 60)
        print("STEP 3: TRAINING RANDOM FOREST MODEL (Model 2)")
        print("=" * 60)
        
        lgb_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        lgb_model.fit(self.X_train, self.y_train)
        
        self.models['lightgbm'] = lgb_model
        
        # Predictions
        y_pred_lgb = lgb_model.predict(self.X_test)
        y_pred_proba_lgb = lgb_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluation
        auc_lgb = roc_auc_score(self.y_test, y_pred_proba_lgb)
        f1_lgb = f1_score(self.y_test, y_pred_lgb)
        
        print(f"Random Forest ROC-AUC: {auc_lgb:.4f}")
        print(f"Random Forest F1-Score: {f1_lgb:.4f}")
        print("\nClassification Report (Random Forest):")
        print(classification_report(self.y_test, y_pred_lgb, target_names=['No Churn', 'Churn']))
        
        return lgb_model, y_pred_lgb, y_pred_proba_lgb
    
    def cross_validate_models(self):
        """Perform cross-validation"""
        print("\n" + "=" * 60)
        print("STEP 4: CROSS-VALIDATION")
        print("=" * 60)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Gradient Boosting CV
        xgb_cv = cross_validate(
            self.models['xgboost'], self.X_train, self.y_train,
            cv=skf, scoring=['roc_auc', 'f1', 'precision', 'recall'],
            n_jobs=-1
        )
        
        print("Gradient Boosting Cross-Validation Results (5-Fold):")
        print(f"  ROC-AUC: {xgb_cv['test_roc_auc'].mean():.4f} (+/- {xgb_cv['test_roc_auc'].std():.4f})")
        print(f"  F1-Score: {xgb_cv['test_f1'].mean():.4f} (+/- {xgb_cv['test_f1'].std():.4f})")
        print(f"  Precision: {xgb_cv['test_precision'].mean():.4f} (+/- {xgb_cv['test_precision'].std():.4f})")
        print(f"  Recall: {xgb_cv['test_recall'].mean():.4f} (+/- {xgb_cv['test_recall'].std():.4f})")
        
        # Random Forest CV
        lgb_cv = cross_validate(
            self.models['lightgbm'], self.X_train, self.y_train,
            cv=skf, scoring=['roc_auc', 'f1', 'precision', 'recall'],
            n_jobs=-1
        )
        
        print("\nRandom Forest Cross-Validation Results (5-Fold):")
        print(f"  ROC-AUC: {lgb_cv['test_roc_auc'].mean():.4f} (+/- {lgb_cv['test_roc_auc'].std():.4f})")
        print(f"  F1-Score: {lgb_cv['test_f1'].mean():.4f} (+/- {lgb_cv['test_f1'].std():.4f})")
        print(f"  Precision: {lgb_cv['test_precision'].mean():.4f} (+/- {lgb_cv['test_precision'].std():.4f})")
        print(f"  Recall: {lgb_cv['test_recall'].mean():.4f} (+/- {lgb_cv['test_recall'].std():.4f})")
    
    def create_evaluation_plots(self, y_pred_proba_xgb, y_pred_proba_lgb):
        """Create model evaluation visualizations"""
        print("\nGenerating evaluation visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Evaluation - Gradient Boosting vs Random Forest', fontsize=16, fontweight='bold')
        
        # ROC Curves
        fpr_xgb, tpr_xgb, _ = roc_curve(self.y_test, y_pred_proba_xgb)
        fpr_lgb, tpr_lgb, _ = roc_curve(self.y_test, y_pred_proba_lgb)
        
        axes[0, 0].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(self.y_test, y_pred_proba_xgb):.3f})', linewidth=2)
        axes[0, 0].plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={roc_auc_score(self.y_test, y_pred_proba_lgb):.3f})', linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confusion Matrices
        cm_xgb = confusion_matrix(self.y_test, self.models['xgboost'].predict(self.X_test))
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
        axes[0, 1].set_title('Confusion Matrix - XGBoost')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        cm_lgb = confusion_matrix(self.y_test, self.models['lightgbm'].predict(self.X_test))
        sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Greens', ax=axes[0, 2], cbar=False)
        axes[0, 2].set_title('Confusion Matrix - LightGBM')
        axes[0, 2].set_ylabel('True Label')
        axes[0, 2].set_xlabel('Predicted Label')
        
        # Feature Importance - Gradient Boosting
        importance_xgb = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.models['xgboost'].feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[1, 0].barh(importance_xgb['feature'], importance_xgb['importance'], color='#3498db')
        axes[1, 0].set_title('Top 10 Features - Gradient Boosting')
        axes[1, 0].set_xlabel('Importance')
        
        # Feature Importance - Random Forest
        importance_lgb = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.models['lightgbm'].feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[1, 1].barh(importance_lgb['feature'], importance_lgb['importance'], color='#2ecc71')
        axes[1, 1].set_title('Top 10 Features - Random Forest')
        axes[1, 1].set_xlabel('Importance')
        
        # Probability Distribution
        axes[1, 2].hist(y_pred_proba_xgb[self.y_test==0], bins=30, alpha=0.6, label='No Churn (GB)', color='#3498db')
        axes[1, 2].hist(y_pred_proba_xgb[self.y_test==1], bins=30, alpha=0.6, label='Churn (GB)', color='#e74c3c')
        axes[1, 2].set_xlabel('Predicted Probability')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Probability Distribution - Gradient Boosting')
        axes[1, 2].legend()
        axes[1, 2].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig('models/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("✓ Model evaluation plots saved to 'model_evaluation.png'")
        plt.close()
    
    def save_models(self):
        """Save trained models"""
        with open('models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.models['xgboost'], f)
        
        with open('models/lightgbm_model.pkl', 'wb') as f:
            pickle.dump(self.models['lightgbm'], f)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        with open('models/feature_cols.pkl', 'wb') as f:
            pickle.dump(self.feature_cols, f)
        
        print("\n✓ Models saved successfully")


if __name__ == "__main__":
    # Initialize
    trainer = ChurnModelTrainer('Data/churn_preprocessed.csv')
    
    # Execute pipeline
    trainer.prepare_data()
    xgb_model, y_pred_xgb, y_pred_proba_xgb = trainer.train_xgboost()
    lgb_model, y_pred_lgb, y_pred_proba_lgb = trainer.train_lightgbm()
    trainer.cross_validate_models()
    trainer.create_evaluation_plots(y_pred_proba_xgb, y_pred_proba_lgb)
    trainer.save_models()
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE ✓")
    print("=" * 60)