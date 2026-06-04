"""
=============================================================================
PROJECT: COVID-19 / Public Health Data Dashboard
ENGINEERING: Pandas (Ingestion), NumPy (Statistics), Matplotlib & Plotly (Viz)
FRAMEWORK: Streamlit Interactive Suite
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- 1. SYSTEM & UI CONFIGURATION ---
st.set_page_config(
    page_title="Public Health Intelligence | COVID-19",
    page_icon="🏥",
    layout="wide"
)

# Advanced CSS for Medical Command Center Aesthetics
st.markdown("""
    <style>
    /* Global Background */
    .stApp { background-color: #050b14; color: #e0e6ed; }
    
    /* Custom Metric Cards with Neon Borders */
    [data-testid="stMetric"] {
        background: #0f172a;
        border: 1px solid #1e293b;
        padding: 20px !important;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Header Container */
    .header-box {
        padding: 30px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border-radius: 15px;
        margin-bottom: 25px;
        border-left: 5px solid #38bdf8;
    }
    
    /* Titles */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #f8fafc; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #050b14; }
    ::-webkit-scrollbar-thumb { background: #1e3a8a; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA PROCESSING CORE (NumPy & Pandas) ---
class HealthDataEngine:
    def __init__(self, path):
        self.path = path
        self.color_map = {
            'Confirmed': '#38bdf8', # Cyan
            'Deaths': '#fb7185',    # Coral/Red
            'Recovered': '#34d399', # Emerald
            'Active': '#fbbf24'     # Amber
        }

    @st.cache_data
    def load_and_stat_analysis(_self):
        if not os.path.exists(_self.path):
            return None
        
        df = pd.read_csv(_self.path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ingestion Cleaning
        cols = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        df[cols] = df[cols].fillna(0).astype(int)
        
        # NumPy Statistical Analysis: Daily Fluctuations
        df = df.sort_values(['Country/Region', 'Province/State', 'Date'])
        
        # Calculate Daily Deltas
        df['New_Infections'] = df.groupby(['Country/Region'])['Confirmed'].diff().fillna(0).clip(lower=0)
        
        # Statistical Rolling Averages (NumPy-based logic)
        df['7Day_Avg'] = df.groupby('Country/Region')['New_Infections'].transform(
            lambda x: x.rolling(window=7).mean()
        ).fillna(0)
        
        # Mortality & Recovery Efficiency (NumPy)
        df['Fatality_Ratio'] = np.where(df['Confirmed'] > 0, (df['Deaths']/df['Confirmed'])*100, 0)
        df['Recovery_Ratio'] = np.where(df['Confirmed'] > 0, (df['Recovered']/df['Confirmed'])*100, 0)
        
        return df

# Initialize Data
engine = HealthDataEngine('covid_19_clean_complete.csv')
master_df = engine.load_and_stat_analysis()

# --- 3. DASHBOARD INTERFACE ---
if master_df is None:
    st.error("🚨 DATABASE MISSING: 'covid_19_clean_complete.csv' not found.")
    st.info("Check your VS Code folder for the dataset file.")
else:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("🛠️ Analysis Suite")
    
    # Navigation
    app_mode = st.sidebar.radio("View Perspective", 
        ["🌐 Global Command", "📊 Deep Statistical Analysis", "📈 Growth Dynamics"])
    
    st.sidebar.markdown("---")
    
    # Interactive Filters
    regions = ["All"] + sorted(master_df['WHO Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("WHO Region", regions)
    
    countries = sorted(master_df['Country/Region'].unique())
    selected_country = st.sidebar.selectbox("Focal Country", ["Global"] + countries)
    
    # Date Slider
    min_date = master_df['Date'].min().to_pydatetime()
    max_date = master_df['Date'].max().to_pydatetime()
    date_range = st.sidebar.date_input("Analysis Window", [min_date, max_date])

    # --- DATA FILTERING ENGINE ---
    df_filtered = master_df.copy()
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered['WHO Region'] == selected_region]
    if selected_country != "Global":
        df_filtered = df_filtered[df_filtered['Country/Region'] == selected_country]
    
    if len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['Date'] >= pd.Timestamp(date_range[0])) & 
                                  (df_filtered['Date'] <= pd.Timestamp(date_range[1]))]

    # --- TOP HEADER ---
    st.markdown(f"""
        <div class="header-box">
            <h1 style='margin:0;'>COVID-19 Public Health Intelligence</h1>
            <p style='margin:0; opacity:0.8;'>Target: {selected_country} | Region: {selected_region} | Mode: {app_mode}</p>
        </div>
    """, unsafe_allow_html=True)

    # Aggregated Stats
    daily_sum = df_filtered.groupby('Date').agg({
        'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum', 'Active': 'sum',
        'New_Infections': 'sum', '7Day_Avg': 'sum'
    }).reset_index()

    # --- 4. DASHBOARD MODES ---
    
    if app_mode == "🌐 Global Command":
        # Metric Grid
        latest = daily_sum.iloc[-1]
        prev = daily_sum.iloc[-2] if len(daily_sum) > 1 else latest
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confirmed Cases", f"{int(latest['Confirmed']):,}", f"{int(latest['Confirmed']-prev['Confirmed']):,} 24h")
        m2.metric("Total Fatalities", f"{int(latest['Deaths']):,}", f"{int(latest['Deaths']-prev['Deaths']):,} 24h", delta_color="inverse")
        m3.metric("Total Recoveries", f"{int(latest['Recovered']):,}")
        m4.metric("Active Burden", f"{int(latest['Active']):,}")

        st.write("---")
        
        # Main Line Chart (Plotly)
        st.subheader("Interactive Temporal Analysis")
        fig_main = go.Figure()
        for col in ['Confirmed', 'Recovered', 'Deaths']:
            fig_main.add_trace(go.Scatter(
                x=daily_sum['Date'], y=daily_sum[col], 
                name=col, line=dict(color=engine.color_map[col], width=3)
            ))
        fig_main.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_main, use_container_width=True)

        # Matplotlib Section (Static Summary Report)
        st.write("---")
        st.subheader("Statistical Summary Report (Matplotlib)")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            # Matplotlib Bar Chart
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#050b14')
            ax.set_facecolor('#0f172a')
            ax.bar(['Recovered', 'Active', 'Deaths'], 
                   [latest['Recovered'], latest['Active'], latest['Deaths']], 
                   color=[engine.color_map['Recovered'], engine.color_map['Active'], engine.color_map['Deaths']])
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_title("Current Case Distribution", color='white', fontsize=14)
            st.pyplot(fig)

        with col_m2:
            # NumPy Correlation or Heatmap logic
            st.info("📊 **Intelligence Insight:** Based on current filtering, the average daily new cases in this window is " + 
                    f"**{int(np.mean(daily_sum['New_Infections'])):,}**. The volatility (Standard Deviation) " + 
                    f"is **{int(np.std(daily_sum['New_Infections'])):,}**.")

    elif app_mode == "📊 Deep Statistical Analysis":
        st.subheader("Recovery Efficiency vs. Mortality Risk")
        
        # Advanced Bubble Map
        map_data = df_filtered[df_filtered['Date'] == df_filtered['Date'].max()]
        fig_map = px.scatter_mapbox(
            map_data, lat="Lat", lon="Long", size="Confirmed", color="Fatality_Ratio",
            hover_name="Country/Region", color_continuous_scale="Reds",
            mapbox_style="carto-darkmatter", zoom=1, height=600
        )
        fig_map.update_layout(template="plotly_dark", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
        # WHO Regional Ranking
        st.write("---")
        st.subheader("WHO Regional Performance Matrix")
        reg_df = map_data.groupby('WHO Region')[['Confirmed', 'Deaths']].sum().reset_index()
        fig_reg = px.bar(reg_df, x='WHO Region', y='Confirmed', color='Deaths', 
                         color_continuous_scale='Turbo', template="plotly_dark")
        st.plotly_chart(fig_reg, use_container_width=True)

    elif app_mode == "📈 Growth Dynamics":
        st.subheader("Epidemiological Curve & Velocity")
        
        # New Cases vs 7-Day Average
        fig_new = go.Figure()
        fig_new.add_trace(go.Bar(x=daily_sum['Date'], y=daily_sum['New_Infections'], 
                                 name="Daily New", marker_color='#38bdf8', opacity=0.4))
        fig_new.add_trace(go.Scatter(x=daily_sum['Date'], y=daily_sum['7Day_Avg'], 
                                     name="7-Day Moving Avg", line=dict(color='#fbbf24', width=2)))
        fig_new.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_new, use_container_width=True)
        
        # Data Export
        st.write("---")
        st.subheader("Export Audit Dataset")
        st.dataframe(df_filtered.head(100), use_container_width=True)
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Data", csv, "covid_export.csv", "text/csv")

# Footer
st.markdown("<center><p style='color: #475569;'>Dashboard Version 2.4 | Designed for Clinical Research Data Ingestion</p></center>", unsafe_allow_html=True)