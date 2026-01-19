import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .churn-no {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_data = joblib.load('churn_model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first using train_model.py")
        return None

def preprocess_input(input_df, feature_names):
    """Preprocess input data to match training features"""
    # Create a DataFrame with all required features initialized to 0
    processed_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Map input values to processed DataFrame
    for col in input_df.columns:
        if col in processed_df.columns:
            processed_df[col] = input_df[col].values[0]
    
    return processed_df

def main():
    # Header
    st.markdown('<h1 class="main-header">📞 Telecom Churn Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Customer Retention Analytics</p>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Analytics"])
    
    # ===========================
    # HOME PAGE
    # ===========================
    if page == "🏠 Home":
        st.write("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>🎯 Objective</h3>
                    <p>Predict customer churn to enable proactive retention strategies</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>🤖 Model Type</h3>
                    <p>Gradient Boosting / Random Forest Classifier</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>📊 Features</h3>
                    <p>17+ customer behavior and usage attributes</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.write("---")
        
        st.subheader("📋 Dataset Features")
        
        features_info = {
            "Feature": [
                "State", "Account Length", "Area Code", "Int'l Plan", "VMail Plan",
                "VMail Message", "Day Mins", "Day Calls", "Day Charge", "Eve Mins",
                "Eve Calls", "Eve Charge", "Night Mins", "Night Calls", "Night Charge",
                "Intl Mins", "Intl Calls", "Intl Charge", "CustServ Calls"
            ],
            "Description": [
                "US state (two-letter code)",
                "Days account has been active",
                "Three-digit area code",
                "Has international plan (yes/no)",
                "Has voice mail plan (yes/no)",
                "Average voice mail messages per month",
                "Total daytime calling minutes",
                "Total daytime calls",
                "Billed cost of daytime calls",
                "Total evening calling minutes",
                "Total evening calls",
                "Billed cost of evening calls",
                "Total nighttime calling minutes",
                "Total nighttime calls",
                "Billed cost of nighttime calls",
                "Total international calling minutes",
                "Total international calls",
                "Billed cost of international calls",
                "Number of customer service calls"
            ],
            "Type": [
                "Categorical", "Numeric", "Categorical", "Binary", "Binary",
                "Numeric", "Numeric", "Numeric", "Numeric", "Numeric",
                "Numeric", "Numeric", "Numeric", "Numeric", "Numeric",
                "Numeric", "Numeric", "Numeric", "Numeric"
            ]
        }
        
        features_df = pd.DataFrame(features_info)
        st.dataframe(features_df, use_container_width=True, hide_index=True)
        
        st.write("---")
        
        st.subheader("🚀 Quick Start Guide")
        st.markdown("""
        1. **Single Prediction**: Navigate to the prediction page to analyze individual customers
        2. **Batch Prediction**: Upload a CSV file to predict churn for multiple customers
        3. **Model Analytics**: View detailed model performance metrics and insights
        
        **Key Insights:**
        - Customer service calls are highly predictive of churn
        - International plan subscribers show different churn patterns
        - Usage patterns (day/evening/night) impact churn likelihood
        """)
    
    # ===========================
    # SINGLE PREDICTION PAGE
    # ===========================
    elif page == "🔮 Single Prediction":
        st.subheader("Predict Churn for Individual Customer")
        st.write("Enter customer information to predict churn probability")
        
        st.write("---")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📍 Location & Account Info")
            state = st.selectbox("State", ['OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 
                                           'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ',
                                           'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR',
                                           'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA',
                                           'NM', 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'KS', 'ND'])
            account_length = st.number_input("Account Length (days)", min_value=1, max_value=300, value=100)
            area_code = st.selectbox("Area Code", ['415', '408', '510'])
            
        with col2:
            st.markdown("#### 📞 Service Plans")
            intl_plan = st.selectbox("International Plan", ["no", "yes"])
            vmail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])
            vmail_message = st.number_input("Voice Mail Messages", min_value=0, max_value=60, value=0)
            custserv_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
        
        with col3:
            st.markdown("#### ⏰ Usage Patterns")
            day_mins = st.number_input("Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
            day_calls = st.number_input("Day Calls", min_value=0, max_value=200, value=100)
            eve_mins = st.number_input("Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
            eve_calls = st.number_input("Evening Calls", min_value=0, max_value=200, value=100)
        
        col4, col5 = st.columns(2)
        
        with col4:
            night_mins = st.number_input("Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
            night_calls = st.number_input("Night Calls", min_value=0, max_value=200, value=100)
        
        with col5:
            intl_mins = st.number_input("International Minutes", min_value=0.0, max_value=30.0, value=10.0)
            intl_calls = st.number_input("International Calls", min_value=0, max_value=30, value=3)
        
        st.write("---")
        
        # Calculate charges (approximation)
        day_charge = day_mins * 0.17
        eve_charge = eve_mins * 0.085
        night_charge = night_mins * 0.045
        intl_charge = intl_mins * 0.27
        
        # Prepare input data
        input_data = {
            'Account Length': account_length,
            "Int'l Plan": 1 if intl_plan == 'yes' else 0,
            'VMail Plan': 1 if vmail_plan == 'yes' else 0,
            'VMail Message': vmail_message,
            'Day Mins': day_mins,
            'Day Calls': day_calls,
            'Day Charge': day_charge,
            'Eve Mins': eve_mins,
            'Eve Calls': eve_calls,
            'Eve Charge': eve_charge,
            'Night Mins': night_mins,
            'Night Calls': night_calls,
            'Night Charge': night_charge,
            'Intl Mins': intl_mins,
            'Intl Calls': intl_calls,
            'Intl Charge': intl_charge,
            'CustServ Calls': custserv_calls
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Add state one-hot encoding
        for s in ['OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'KS', 'ND']:
            input_df[f'State_{s}'] = 1 if state == s else 0
        
        # Add area code one-hot encoding
        for ac in ['415', '408', '510']:
            input_df[f'AreaCode_{ac}'] = 1 if area_code == ac else 0
        
        # Add engineered features
        input_df['Avg_Day_Call_Duration'] = day_mins / (day_calls + 1)
        input_df['Avg_Eve_Call_Duration'] = eve_mins / (eve_calls + 1)
        input_df['Avg_Night_Call_Duration'] = night_mins / (night_calls + 1)
        input_df['Total_Charge'] = day_charge + eve_charge + night_charge + intl_charge
        input_df['Total_Mins'] = day_mins + eve_mins + night_mins + intl_mins
        input_df['Total_Calls'] = day_calls + eve_calls + night_calls + intl_calls
        
        # Predict button
        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            # Preprocess input
            processed_input = preprocess_input(input_df, feature_names)
            
            # Scale features
            input_scaled = scaler.transform(processed_input)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            churn_prob = probability[1] * 100
            no_churn_prob = probability[0] * 100
            
            st.write("---")
            
            # Display results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-box churn-yes">
                            ⚠️ HIGH RISK<br>
                            Customer Likely to Churn
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-box churn-no">
                            ✅ LOW RISK<br>
                            Customer Likely to Stay
                        </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob,
                    title={'text': "Churn Probability"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if churn_prob > 50 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
            
            # Detailed probabilities
            st.subheader("Prediction Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric("Probability of Churn", f"{churn_prob:.2f}%")
            with prob_col2:
                st.metric("Probability of Retention", f"{no_churn_prob:.2f}%")
            
            # Recommendations
            st.write("---")
            st.subheader("💡 Recommendations")
            
            if prediction == 1:
                recommendations = []
                
                if custserv_calls > 3:
                    recommendations.append("🔴 **High customer service calls detected** - Immediate follow-up required to address concerns")
                
                if intl_plan == 'yes':
                    recommendations.append("🌍 **International plan subscriber** - Consider offering competitive international rates")
                
                if day_charge + eve_charge + night_charge > 50:
                    recommendations.append("💰 **High usage charges** - Offer loyalty discount or optimized plan")
                
                if vmail_plan == 'no':
                    recommendations.append("📧 **No voice mail plan** - Offer complementary voice mail trial")
                
                if len(recommendations) == 0:
                    recommendations.append("📞 **Proactive outreach** - Contact customer to ensure satisfaction")
                
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("✅ Customer appears satisfied. Continue monitoring and maintain service quality.")
    
    # ===========================
    # BATCH PREDICTION PAGE
    # ===========================
    elif page == "📊 Batch Prediction":
        st.subheader("Batch Prediction for Multiple Customers")
        st.write("Upload a CSV file with customer data to predict churn for multiple customers")
        
        st.write("---")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! {len(df)} customers found.")
            
            # Show preview
            with st.expander("📋 View Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Process data similar to training
                    df_processed = df.copy()
                    
                    # Drop Phone if exists
                    if 'Phone' in df_processed.columns:
                        df_processed = df_processed.drop('Phone', axis=1)
                    
                    # Drop target if exists
                    if 'Churn?' in df_processed.columns:
                        actual_churn = df_processed['Churn?'].copy()
                        df_processed = df_processed.drop('Churn?', axis=1)
                    else:
                        actual_churn = None
                    
                    # Encode binary variables
                    binary_cols = ['Int\'l Plan', 'VMail Plan']
                    for col in binary_cols:
                        if col in df_processed.columns:
                            df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
                    
                    # One-hot encode State
                    if 'State' in df_processed.columns:
                        df_processed = pd.get_dummies(df_processed, columns=['State'], prefix='State')
                    
                    # One-hot encode Area Code
                    if 'Area Code' in df_processed.columns:
                        df_processed = pd.get_dummies(df_processed, columns=['Area Code'], prefix='AreaCode')
                    
                    # Create engineered features
                    if 'Day Mins' in df_processed.columns and 'Day Calls' in df_processed.columns:
                        df_processed['Avg_Day_Call_Duration'] = df_processed['Day Mins'] / (df_processed['Day Calls'] + 1)
                    
                    if 'Eve Mins' in df_processed.columns and 'Eve Calls' in df_processed.columns:
                        df_processed['Avg_Eve_Call_Duration'] = df_processed['Eve Mins'] / (df_processed['Eve Calls'] + 1)
                    
                    if 'Night Mins' in df_processed.columns and 'Night Calls' in df_processed.columns:
                        df_processed['Avg_Night_Call_Duration'] = df_processed['Night Mins'] / (df_processed['Night Calls'] + 1)
                    
                    charge_cols = [c for c in df_processed.columns if 'Charge' in c]
                    if charge_cols:
                        df_processed['Total_Charge'] = df_processed[charge_cols].sum(axis=1)
                    
                    mins_cols = [c for c in df_processed.columns if 'Mins' in c]
                    if mins_cols:
                        df_processed['Total_Mins'] = df_processed[mins_cols].sum(axis=1)
                    
                    calls_cols = [c for c in df_processed.columns if 'Calls' in c and c != 'CustServ Calls']
                    if calls_cols:
                        df_processed['Total_Calls'] = df_processed[calls_cols].sum(axis=1)
                    
                    # Align with model features
                    for col in feature_names:
                        if col not in df_processed.columns:
                            df_processed[col] = 0
                    
                    df_processed = df_processed[feature_names]
                    
                    # Scale and predict
                    df_scaled = scaler.transform(df_processed)
                    predictions = model.predict(df_scaled)
                    probabilities = model.predict_proba(df_scaled)[:, 1]
                    
                    # Add predictions to original dataframe
                    df['Churn_Prediction'] = predictions
                    df['Churn_Prediction'] = df['Churn_Prediction'].map({1: 'Yes', 0: 'No'})
                    df['Churn_Probability'] = (probabilities * 100).round(2)
                    df['Risk_Level'] = pd.cut(probabilities, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
                    
                st.success("Predictions completed!")
                
                # Summary metrics
                st.write("---")
                st.subheader("📊 Prediction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                churn_count = (predictions == 1).sum()
                churn_rate = (churn_count / len(predictions)) * 100
                
                with col1:
                    st.metric("Total Customers", len(df))
                with col2:
                    st.metric("Predicted Churners", churn_count)
                with col3:
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                with col4:
                    avg_risk = probabilities.mean() * 100
                    st.metric("Average Risk", f"{avg_risk:.1f}%")
                
                # Visualizations
                st.write("---")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Churn distribution
                    churn_dist = df['Churn_Prediction'].value_counts()
                    fig = px.pie(values=churn_dist.values, names=churn_dist.index, 
                                title="Churn Prediction Distribution",
                                color_discrete_map={'Yes': '#ef4444', 'No': '#10b981'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_col2:
                    # Risk level distribution
                    risk_dist = df['Risk_Level'].value_counts().sort_index()
                    fig = px.bar(x=risk_dist.index, y=risk_dist.values,
                                title="Risk Level Distribution",
                                labels={'x': 'Risk Level', 'y': 'Count'},
                                color=risk_dist.index,
                                color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.write("---")
                st.subheader("📋 Detailed Results")
                
                # Filter options
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    filter_churn = st.multiselect("Filter by Churn Prediction", 
                                                 options=['Yes', 'No'], 
                                                 default=['Yes', 'No'])
                
                with filter_col2:
                    filter_risk = st.multiselect("Filter by Risk Level", 
                                                options=['Low', 'Medium', 'High'], 
                                                default=['Low', 'Medium', 'High'])
                
                filtered_df = df[df['Churn_Prediction'].isin(filter_churn) & df['Risk_Level'].isin(filter_risk)]
                
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                
                # Download results
                st.write("---")
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # ===========================
    # MODEL ANALYTICS PAGE
    # ===========================
    elif page == "📈 Model Analytics":
        st.subheader("Model Performance & Analytics")
        
        st.write("---")
        
        # Model info
        st.markdown("### 🤖 Model Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.info(f"**Algorithm:** {type(model).__name__}")
        with info_col2:
            st.info(f"**Features:** {len(feature_names)}")
        with info_col3:
            st.info("**Task:** Binary Classification")
        
        st.write("---")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(feature_importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="Top 15 Most Important Features",
                        color='Importance',
                        color_continuous_scale='Blues')
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
        
        # Key insights
        st.markdown("### 💡 Key Model Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            **High-Risk Indicators:**
            - 🔴 Customer service calls > 3
            - 🔴 International plan subscribers
            - 🔴 High usage charges (>$50/month)
            - 🔴 No voice mail plan
            - 🔴 Low account length (<30 days)
            """)
        
        with insights_col2:
            st.markdown("""
            **Retention Factors:**
            - 🟢 Consistent usage patterns
            - 🟢 Voice mail plan subscribers
            - 🟢 Low customer service interactions
            - 🟢 Moderate charges ($20-$40)
            - 🟢 Longer account tenure (>180 days)
            """)
        
        st.write("---")
        
        # Business recommendations
        st.markdown("### 📊 Business Recommendations")
        
        st.markdown("""
        1. **Proactive Outreach**: Contact customers with >3 customer service calls immediately
        2. **Plan Optimization**: Review and optimize pricing for international plans
        3. **Loyalty Programs**: Implement retention programs for high-value customers at risk
        4. **Service Quality**: Focus on first-call resolution to reduce repeat service calls
        5. **Predictive Monitoring**: Use this model for daily churn risk scoring
        6. **A/B Testing**: Test retention strategies on medium-risk customers first
        """)

if __name__ == "__main__":
    main()