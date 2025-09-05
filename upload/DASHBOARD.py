import streamlit as st
import math
import random

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
    .chart-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data using only standard Python
def generate_sample_data():
    random.seed(42)
    sample_size = 250
    
    data = {
        'student_id': list(range(1, sample_size + 1)),
        'attendance_rate': [random.uniform(0.5, 1.0) for _ in range(sample_size)],
        'avg_grade': [random.uniform(60, 95) for _ in range(sample_size)],
        'hours_studied_per_week': [random.randint(5, 40) for _ in range(sample_size)],
        'sleep_hours': [random.uniform(4, 10) for _ in range(sample_size)],
        'exercise_per_week': [random.randint(0, 7) for _ in range(sample_size)],
        'faculty': [random.choice(["Science", "Arts", "Business", "Engineering", "Medicine"]) for _ in range(sample_size)]
    }
    
    # Create risk scores and status
    risk_scores = []
    risk_statuses = []
    
    for i in range(sample_size):
        risk_score = (
            (1 - data['attendance_rate'][i]) * 0.3 +
            ((100 - data['avg_grade'][i]) / 100) * 0.3 +
            (0.2 if data['sleep_hours'][i] < 6 else 0) +
            (0.2 if data['exercise_per_week'][i] < 2 else 0)
        )
        risk_scores.append(risk_score)
        risk_statuses.append('At Risk 🚨' if risk_score > 0.5 else 'Safe ✅')
    
    data['risk_score'] = risk_scores
    data['risk_status'] = risk_statuses
    
    return data

# Create a simple bar chart using Streamlit's native functions
def st_bar_chart(data, title, x_label, y_label):
    chart_data = {}
    for key, value in data.items():
        chart_data[key] = value
    
    st.write(f"**{title}**")
    st.bar_chart(chart_data)

# Create a simple pie chart using Streamlit's native functions
def st_pie_chart(data, title):
    st.write(f"**{title}**")
    
    # Create a simple text-based representation
    total = sum(data.values())
    for label, value in data.items():
        percentage = (value / total) * 100
        st.write(f"{label}: {value} ({percentage:.1f}%)")
        
        # Create a simple bar to represent the percentage
        st.progress(percentage/100, text=f"{percentage:.1f}%")

# Dashboard Header
st.markdown('<h1 class="dashboard-header">📊 Student Analytics Dashboard</h1>', unsafe_allow_html=True)

# Generate data
data = generate_sample_data()

# Key Metrics
st.markdown("### Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    at_risk_count = data['risk_status'].count('At Risk 🚨')
    at_risk_percent = at_risk_count / len(data['risk_status']) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">At-Risk Students</div>
        <div class="metric-value">{at_risk_count}</div>
        <div>{at_risk_percent:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_grade = sum(data['avg_grade']) / len(data['avg_grade'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Grade</div>
        <div class="metric-value">{avg_grade:.1f}</div>
        <div>out of 100</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_sleep = sum(data['sleep_hours']) / len(data['sleep_hours'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Sleep Hours</div>
        <div class="metric-value">{avg_sleep:.1f}</div>
        <div>per night</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_study = sum(data['hours_studied_per_week']) / len(data['hours_studied_per_week'])
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
    risk_counts = {
        'At Risk 🚨': data['risk_status'].count('At Risk 🚨'),
        'Safe ✅': data['risk_status'].count('Safe ✅')
    }
    st_pie_chart(risk_counts, 'Student Risk Distribution')

with col2:
    # Average Metrics by Faculty
    faculty_data = {}
    for i, faculty in enumerate(data['faculty']):
        if faculty not in faculty_data:
            faculty_data[faculty] = {'grades': [], 'study_hours': []}
        
        faculty_data[faculty]['grades'].append(data['avg_grade'][i])
        faculty_data[faculty]['study_hours'].append(data['hours_studied_per_week'][i])
    
    avg_grades = {faculty: sum(values['grades'])/len(values['grades']) for faculty, values in faculty_data.items()}
    avg_study_hours = {faculty: sum(values['study_hours'])/len(values['study_hours']) for faculty, values in faculty_data.items()}
    
    st.write("**Average Grade by Faculty**")
    st.bar_chart(avg_grades)
    
    st.write("**Average Study Hours by Faculty**")
    st.bar_chart(avg_study_hours)

# Additional Charts
st.markdown('<div class="section-header">Student Behavior Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Sleep distribution
    sleep_ranges = {
        'Less than 6 hrs': len([h for h in data['sleep_hours'] if h < 6]),
        '6-7 hrs': len([h for h in data['sleep_hours'] if 6 <= h < 7]),
        '7-8 hrs': len([h for h in data['sleep_hours'] if 7 <= h < 8]),
        '8+ hrs': len([h for h in data['sleep_hours'] if h >= 8])
    }
    st_bar_chart(sleep_ranges, 'Sleep Distribution', 'Hours', 'Number of Students')

with col2:
    # Study hours distribution
    study_ranges = {
        'Less than 10 hrs': len([h for h in data['hours_studied_per_week'] if h < 10]),
        '10-20 hrs': len([h for h in data['hours_studied_per_week'] if 10 <= h < 20]),
        '20-30 hrs': len([h for h in data['hours_studied_per_week'] if 20 <= h < 30]),
        '30+ hrs': len([h for h in data['hours_studied_per_week'] if h >= 30])
    }
    st_bar_chart(study_ranges, 'Study Hours Distribution', 'Hours', 'Number of Students')

# Faculty-wise Risk Analysis
st.markdown('<div class="section-header">Faculty-wise Analysis</div>', unsafe_allow_html=True)

# Calculate risk by faculty
faculty_risk = {}
for faculty in set(data['faculty']):
    faculty_risk[faculty] = {
        'At Risk 🚨': 0,
        'Safe ✅': 0
    }

for i, faculty in enumerate(data['faculty']):
    faculty_risk[faculty][data['risk_status'][i]] += 1

# Display faculty risk data
for faculty, risks in faculty_risk.items():
    total = risks['At Risk 🚨'] + risks['Safe ✅']
    risk_percentage = (risks['At Risk 🚨'] / total) * 100 if total > 0 else 0
    
    st.write(f"**{faculty}**")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"At Risk: {risks['At Risk 🚨']} students")
        st.progress(risk_percentage/100, text=f"{risk_percentage:.1f}% at risk")
    
    with col2:
        st.write(f"Safe: {risks['Safe ✅']} students")

# Data Table
st.markdown('<div class="section-header">Student Data Overview</div>', unsafe_allow_html=True)

# Create a sample table with limited data for display
sample_size = min(10, len(data['student_id']))
table_data = {
    'Student ID': data['student_id'][:sample_size],
    'Faculty': data['faculty'][:sample_size],
    'Attendance': [f"{x:.0%}" for x in data['attendance_rate'][:sample_size]],
    'Avg Grade': [f"{x:.1f}" for x in data['avg_grade'][:sample_size]],
    'Study Hours': data['hours_studied_per_week'][:sample_size],
    'Sleep Hours': [f"{x:.1f}" for x in data['sleep_hours'][:sample_size]],
    'Risk Status': data['risk_status'][:sample_size]
}

# Display the table
st.table(table_data)

# Additional information
st.markdown("---")
st.info("""
This dashboard shows analytics based on sample student data. Key features include:
- Overview metrics of student performance and well-being
- Risk status distribution across the student population
- Faculty-wise comparison of academic performance
- Analysis of student sleep patterns and study habits
""")

# Add a refresh button to generate new sample data
if st.button("Generate New Sample Data"):
    st.experimental_rerun()
