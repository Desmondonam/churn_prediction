"""
Churn Analysis - Exploratory Data Analysis & Preprocessing
This module performs comprehensive EDA and data preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

class ChurnEDAPreprocessor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.preprocessed_df = None
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_and_clean(self):
        """Load and clean the dataset"""
        print("=" * 60)
        print("STEP 1: DATA LOADING & CLEANING")
        print("=" * 60)
        
        # Clean churn column
        self.df['Churn?'] = self.df['Churn?'].astype(str).str.strip()
        self.df['Churn'] = (self.df['Churn?'] == 'True.').astype(int)
        
        # Remove unnecessary columns
        self.df = self.df.drop(['Phone', 'Churn?'], axis=1)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        print(f"\nChurn Distribution:\n{self.df['Churn'].value_counts()}")
        print(f"Churn Rate: {self.df['Churn'].mean():.2%}")
        
        return self.df
    
    def eda_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "=" * 60)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Statistical summary
        print("\nNumerical Features Summary:")
        print(self.df.describe())
        
        # Churn by State
        print("\nChurn Rate by State (Top 10):")
        state_churn = self.df.groupby('State')['Churn'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
        state_churn.columns = ['Churn_Count', 'Total_Customers', 'Churn_Rate']
        print(state_churn.head(10))
        
        # Churn by Plans
        print("\nChurn Rate by International Plan:")
        print(self.df.groupby("Int'l Plan")['Churn'].agg(['sum', 'count', 'mean']))
        
        print("\nChurn Rate by VoiceMail Plan:")
        print(self.df.groupby("VMail Plan")['Churn'].agg(['sum', 'count', 'mean']))
        
        # Customer Service Calls impact
        print("\nChurn Rate by Customer Service Calls:")
        cs_churn = self.df.groupby('CustServ Calls')['Churn'].agg(['sum', 'count', 'mean'])
        cs_churn.columns = ['Churn_Count', 'Total_Customers', 'Churn_Rate']
        print(cs_churn)
        
        # Cohort Analysis
        print("\nCohort Analysis - Account Length Quartiles:")
        self.df['Account_Length_Quartile'] = pd.qcut(self.df['Account Length'], q=4, labels=['Q1_New', 'Q2_Developing', 'Q3_Mature', 'Q4_Loyal'])
        cohort = self.df.groupby('Account_Length_Quartile')['Churn'].agg(['sum', 'count', 'mean'])
        cohort.columns = ['Churn_Count', 'Total_Customers', 'Churn_Rate']
        print(cohort)
        
        # Retention Cohorts (by account length)
        print("\nRetention by Account Length (months):")
        retention = self.df.groupby(pd.cut(self.df['Account Length'], bins=6))['Churn'].agg(['sum', 'count', lambda x: (1-x.mean())]).round(3)
        retention.columns = ['Churned', 'Total', 'Retention_Rate']
        print(retention)
        
        return state_churn, cohort
    
    def create_visualizations(self):
        """Create EDA visualizations"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Churn Analysis - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0, 0].bar(['No Churn', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Churn Distribution')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(churn_counts.values):
            axes[0, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # 2. Churn by Plan
        plan_data = pd.DataFrame({
            'Intl Plan': self.df.groupby("Int'l Plan")['Churn'].mean(),
            'VMail Plan': self.df.groupby("VMail Plan")['Churn'].mean()
        })
        plan_data.plot(kind='bar', ax=axes[0, 1], color=['#3498db', '#9b59b6'])
        axes[0, 1].set_title('Churn Rate by Plan')
        axes[0, 1].set_ylabel('Churn Rate')
        axes[0, 1].set_xlabel('')
        axes[0, 1].legend(loc='best')
        axes[0, 1].axhline(y=self.df['Churn'].mean(), color='r', linestyle='--', label='Overall Rate')
        
        # 3. Customer Service Calls Impact
        cs_data = self.df.groupby('CustServ Calls')['Churn'].agg(['mean', 'count'])
        axes[0, 2].plot(cs_data.index, cs_data['mean'], marker='o', color='#e67e22', linewidth=2, markersize=8)
        axes[0, 2].set_title('Churn Rate by Customer Service Calls')
        axes[0, 2].set_xlabel('Customer Service Calls')
        axes[0, 2].set_ylabel('Churn Rate')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Day Minutes Distribution by Churn
        axes[1, 0].hist(self.df[self.df['Churn']==0]['Day Mins'], bins=30, alpha=0.6, label='No Churn', color='#2ecc71')
        axes[1, 0].hist(self.df[self.df['Churn']==1]['Day Mins'], bins=30, alpha=0.6, label='Churn', color='#e74c3c')
        axes[1, 0].set_title('Day Minutes Distribution by Churn')
        axes[1, 0].set_xlabel('Day Minutes')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 5. Total Charge vs Churn
        axes[1, 1].scatter(self.df[self.df['Churn']==0]['Day Charge'], 
                          self.df[self.df['Churn']==0]['Eve Charge'],
                          alpha=0.4, s=30, label='No Churn', color='#2ecc71')
        axes[1, 1].scatter(self.df[self.df['Churn']==1]['Day Charge'], 
                          self.df[self.df['Churn']==1]['Eve Charge'],
                          alpha=0.4, s=30, label='Churn', color='#e74c3c')
        axes[1, 1].set_title('Day Charge vs Eve Charge')
        axes[1, 1].set_xlabel('Day Charge ($)')
        axes[1, 1].set_ylabel('Eve Charge ($)')
        axes[1, 1].legend()
        
        # 6. Account Length Cohort
        cohort_data = self.df.groupby('Account_Length_Quartile')['Churn'].agg(['mean', 'count'])
        axes[1, 2].bar(range(len(cohort_data)), cohort_data['mean'].values, color=['#1abc9c', '#3498db', '#f39c12', '#e74c3c'])
        axes[1, 2].set_xticks(range(len(cohort_data)))
        axes[1, 2].set_xticklabels(cohort_data.index, rotation=45)
        axes[1, 2].set_title('Churn Rate by Account Maturity')
        axes[1, 2].set_ylabel('Churn Rate')
        for i, v in enumerate(cohort_data['mean'].values):
            axes[1, 2].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../eda_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ EDA visualizations saved to 'eda_analysis.png'")
        plt.close()
    
    def preprocess_data(self):
        """Preprocess data for modeling"""
        print("\n" + "=" * 60)
        print("STEP 3: DATA PREPROCESSING FOR MODELING")
        print("=" * 60)
        
        df = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['State', "Int'l Plan", "VMail Plan"]
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.encoders[col].fit_transform(df[col])
            else:
                df[col + '_encoded'] = self.encoders[col].transform(df[col])
        
        # Drop original categorical columns and area code
        df = df.drop(['State', "Int'l Plan", "VMail Plan", 'Area Code', 'Account_Length_Quartile'], axis=1)
        
        # Features for modeling
        self.feature_cols = [col for col in df.columns if col not in ['Churn']]
        
        print(f"Features for modeling: {len(self.feature_cols)}")
        print(f"Features: {self.feature_cols}")
        
        self.preprocessed_df = df
        return df, self.feature_cols
    
    def save_preprocessed_data(self, output_path):
        """Save preprocessed data"""
        self.preprocessed_df.to_csv(output_path, index=False)
        print(f"\n✓ Preprocessed data saved to '{output_path}'")
        
        # Save encoders info
        encoders_info = {
            'categorical_cols': ['State', "Int'l Plan", "VMail Plan"],
            'feature_cols': self.feature_cols
        }
        return encoders_info


if __name__ == "__main__":
    # Initialize
    path = r"C:\Users\Admin\Desktop\Freelance\Denis\Projects\Retention_Engine\Data\churn.csv"
    processor = ChurnEDAPreprocessor(path)
    
    # Execute pipeline
    processor.load_and_clean()
    processor.eda_analysis()
    processor.create_visualizations()
    processor.preprocess_data()
    processor.save_preprocessed_data('Data/churn_preprocessed.csv')
    
    print("\n" + "=" * 60)
    print("EDA & PREPROCESSING COMPLETE ✓")
    print("=" * 60)