import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn EDA Dashboard", layout="wide")

st.title("📞 Customer Churn Interactive Analytics")
st.markdown("Explore the drivers behind customer attrition.")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload your Churn CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing
    df['Churn'] = df['Churn?'].astype(str) # For better labeling in charts
    
    # --- Metrics Section ---
    total_cust = len(df)
    churn_rate = (df['Churn'] == 'True.').mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_cust)
    col2.metric("Churn Rate", f"{churn_rate:.2%}")
    col3.metric("Avg Cust Service Calls", round(df['CustServ Calls'].mean(), 2))

    # --- Analysis Tabs ---
    tab1, tab2, tab3 = st.tabs(["Demographics", "Usage Patterns", "Service Issues"])

    with tab1:
        st.subheader("Churn by State & Plan")
        fig_state = px.histogram(df, x="State", color="Churn", barmode="group", 
                                 title="Customer Status by State")
        st.plotly_chart(fig_state, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig_intl = px.sunburst(df, path=['Int\'l Plan', 'Churn'], title="International Plan vs Churn")
            st.plotly_chart(fig_intl)
        with col_b:
            fig_vmail = px.sunburst(df, path=['VMail Plan', 'Churn'], title="VMail Plan vs Churn")
            st.plotly_chart(fig_vmail)

    with tab2:
        st.subheader("Usage Distribution")
        feature = st.selectbox("Select usage metric to analyze:", 
                              ['Day Mins', 'Eve Mins', 'Night Mins', 'Intl Mins'])
        fig_dist = px.box(df, x="Churn", y=feature, color="Churn", points="all",
                         title=f"Distribution of {feature} by Churn Status")
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        st.subheader("The 'Smoking Gun': Customer Service")
        st.write("Does calling customer service frequently indicate a high risk of churn?")
        fig_cust_serv = px.histogram(df, x="CustServ Calls", color="Churn", barmode="group")
        st.plotly_chart(fig_cust_serv, use_container_width=True)

else:
    st.info("Please upload a CSV file to get started.")