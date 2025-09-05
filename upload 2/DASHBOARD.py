import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Student Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .dashboard-header {
        font-size: 2.5rem;
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .metric-title {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    sample_size = 250
    
    data = {
        'student_id': range(1, sample_size + 1),
        'attendance_rate': np.random.uniform(0.5, 1.0, sample_size),
        'avg_grade': np.random.uniform(60, 95, sample_size),
        'hours_studied_per_week': np.random.randint(5, 40, sample_size),
        'sleep_hours': np.random.uniform(4, 10, sample_size),
        'exercise_per_week': np.random.randint(0, 7, sample_size),
        'faculty': np.random.choice(["Science", "Arts", "Business", "Engineering", "Medicine"], sample_size)
    }
    
    df = pd.DataFrame(data)
    
    # Create a mock risk prediction based on some rules
    df['risk_score'] = (
        (1 - df['attendance_rate']) * 0.3 +
        ((100 - df['avg_grade']) / 100) * 0.3 +
        (df['sleep_hours'] < 6).astype(int) * 0.2 +
        (df['exercise_per_week'] < 2).astype(int) * 0.2
    )
    
    df['risk_status'] = df['risk_score'].apply(lambda x: 'At Risk 🚨' if x > 0.5 else 'Safe ✅')
    
    return df

# Load data
df = generate_sample_data()

# Dashboard Header
st.markdown('<h1 class="dashboard-header">📊 Student Analytics Dashboard</h1>', unsafe_allow_html=True)

# Key Metrics
st.markdown("### Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    at_risk_count = len(df[df['risk_status'] == 'At Risk 🚨'])
    at_risk_percent = at_risk_count / len(df) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">At-Risk Students</div>
        <div class="metric-value">{at_risk_count}</div>
        <div>{at_risk_percent:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_grade = df['avg_grade'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Grade</div>
        <div class="metric-value">{avg_grade:.1f}</div>
        <div>out of 100</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_sleep = df['sleep_hours'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Sleep Hours</div>
        <div class="metric-value">{avg_sleep:.1f}</div>
        <div>per night</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_study = df['hours_studied_per_week'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Study Hours</div>
        <div class="metric-value">{avg_study:.1f}</div>
        <div>per week</div>
    </div>
    """, unsafe_allow_html=True)

# Charts Section
st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    # Risk Status Pie Chart
    risk_counts = df['risk_status'].value_counts()
    fig = px.pie(
        values=risk_counts.values, 
        names=risk_counts.index,
        title='Student Risk Distribution',
        color=risk_counts.index,
        color_discrete_map={'At Risk 🚨': '#FF4B4B', 'Safe ✅': '#2ECC71'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Average Metrics by Faculty
    faculty_metrics = df.groupby('faculty').agg({
        'avg_grade': 'mean',
        'sleep_hours': 'mean',
        'hours_studied_per_week': 'mean',
        'risk_score': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Average Grade',
        x=faculty_metrics['faculty'],
        y=faculty_metrics['avg_grade'],
        marker_color='#1f77b4'
    ))
    fig.add_trace(go.Bar(
        name='Study Hours',
        x=faculty_metrics['faculty'],
        y=faculty_metrics['hours_studied_per_week'],
        marker_color='#ff7f0e'
    ))
    fig.update_layout(
        title='Academic Performance by Faculty',
        barmode='group',
        yaxis_title='Value'
    )
    st.plotly_chart(fig, use_container_width=True)

# Additional Charts
st.markdown('<div class="section-header">Student Behavior Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Sleep vs Grade Scatter Plot
    fig = px.scatter(
        df, 
        x='sleep_hours', 
        y='avg_grade',
        color='risk_status',
        color_discrete_map={'At Risk 🚨': '#FF4B4B', 'Safe ✅': '#2ECC71'},
        title='Sleep Hours vs Academic Performance',
        labels={'sleep_hours': 'Sleep Hours per Night', 'avg_grade': 'Average Grade'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Study Hours Distribution by Risk Status
    fig = px.box(
        df, 
        x='risk_status', 
        y='hours_studied_per_week',
        color='risk_status',
        color_discrete_map={'At Risk 🚨': '#FF4B4B', 'Safe ✅': '#2ECC71'},
        title='Study Hours Distribution by Risk Status',
        labels={'risk_status': 'Risk Status', 'hours_studied_per_week': 'Study Hours per Week'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Faculty-wise Risk Analysis
st.markdown('<div class="section-header">Faculty-wise Analysis</div>', unsafe_allow_html=True)

faculty_risk = df.groupby('faculty')['risk_status'].value_counts().unstack(fill_value=0)
faculty_risk_percent = faculty_risk.div(faculty_risk.sum(axis=1), axis=0) * 100

fig = go.Figure()
for status in faculty_risk.columns:
    fig.add_trace(go.Bar(
        name=status,
        x=faculty_risk_percent.index,
        y=faculty_risk_percent[status],
        hovertemplate='<b>%{x}</b><br>%{y:.1f}% ' + status
    ))

fig.update_layout(
    title='Risk Status Distribution by Faculty',
    barmode='stack',
    yaxis_title='Percentage',
    xaxis_title='Faculty'
)

st.plotly_chart(fig, use_container_width=True)

# Data Table
st.markdown('<div class="section-header">Student Data Overview</div>', unsafe_allow_html=True)

# Filter options
st.subheader("Filter Data")
faculty_filter = st.multiselect(
    "Select Faculties:",
    options=df['faculty'].unique(),
    default=df['faculty'].unique()
)

risk_filter = st.multiselect(
    "Select Risk Status:",
    options=df['risk_status'].unique(),
    default=df['risk_status'].unique()
)

# Apply filters
filtered_df = df[
    (df['faculty'].isin(faculty_filter)) & 
    (df['risk_status'].isin(risk_filter))
]

# Show filtered data
st.dataframe(
    filtered_df.drop('student_id', axis=1).style.format({
        'attendance_rate': '{:.0%}',
        'avg_grade': '{:.1f}',
        'sleep_hours': '{:.1f}',
        'risk_score': '{:.3f}'
    }),
    use_container_width=True,
    height=300
)

# Download button
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_student_data.csv",
    mime="text/csv"
)