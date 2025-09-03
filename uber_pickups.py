import streamlit as st
import pandas as pd
import numpy as np 


st.title("Uber pickups app")

#st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Done! (using st.cache_data)')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

if st.checkbox('SHOW row data'):
    st.subheader('Raw data')
    st.write(data)
    st.checkbox
elif st.checkbox('Number of pickups by hours'):
    st.subheader('Number of pickups by hours')
    hist_values = np.histogram(
        data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)
elif st.checkbox('Map of all pickups'):
    hour_to_filter = st.slider('hour', 0 ,23 , 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader(f'Map of all pickups at {hour_to_filter}:00')
    st.map(filtered_data)

# st.subheader('test')
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Temperature", "70 °F", "1.2 °F")
#     st.checkbox('Number of pickups by hourss')
# with col2:
#     st.metric("Wind", "9 mph", "-8%")
# with col3:
#     st.metric("Humidity", "86%", "4%")

# st.subheader('test 2')
# st.caption("dbcidbicsdibiwbifdciw")
