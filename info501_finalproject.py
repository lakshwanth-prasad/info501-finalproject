#Imports 
import streamlit as st 
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo 
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from lazypredict.Supervised import LazyClassifier
from sklearn.semi_supervised import LabelSpreading


@st.cache(allow_output_mutation=True) 
def load_data():
    data = pd.read_csv("healthcare_dataset.csv")
    return data

def create_histogram(data, column):
    fig = px.histogram(data, x=column)
    return fig

def main():
    st.title('Healthcare Data Analysis App')
    st.markdown("An interactive visualization & Prediction of hospital dataset over different categories.")
    st.markdown("Team Members:")
    st.markdown("Lakshwanth Prasad Kothandaraman")
    st.markdown("Neha Dudhane")
    st.markdown("Dhivakar Ramesh")
    st.markdown("Sonam Chhatani")
    data = load_data()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Dataset Overview", "Visualizations","Comparision Visualizations","Trends Visualizations","Machine Learning Model Training & Prediction","Machine Learning MSE"])
    cols = ['Gender','Blood Type', 'Medical Condition',
        'Insurance Provider', 'Admission Type',
        'Medication', 'Test Results','Doctor', 'Hospital']
    data['Date of Admission']= pd.to_datetime(data['Date of Admission'])
    data['Discharge Date']= pd.to_datetime(data['Discharge Date'])
    data['Days hospitalized'] = (data['Discharge Date'] - data['Date of Admission'])
    data['Days hospitalized'] = data['Days hospitalized'].dt.total_seconds() / 86400
    data['Admission Year'] = data['Date of Admission'].dt.year

    if page == "Dataset Overview":
        st.subheader("Dataset Overview")
        st.subheader('Basic Information')
        st.write('First Five Rows of the Dataset:')
        st.dataframe(data.head().T)

        
        st.write('Dataset Information:')
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write('Null Values in Dataset:')
        st.dataframe(data.isna().sum())

        st.write('Statistical Summary of the Dataset:')
        st.dataframe(data.describe())
        st.write('Shape of the Dataset:', data.shape)

    # Exploring numerical features
        st.subheader('Numerical Features')
        st.dataframe(data.describe(exclude=['O']))

    elif page == "Visualizations":
        st.subheader("Data Visualizations")
        viz_type = st.selectbox("Select Visualization Based On", ["Gender Distribution", "Blood Type", "Medical Condition", "Insurance Provider", "Admission Type", "Medication", "Test Results"])    

        if viz_type == "Gender Distribution":
            column = "Gender"  # or use st.selectbox("Select Column", data.columns) for a dynamic choice
            color_map = {"Male": "blue", "Female": "pink"}  # Define the color map
            fig = px.histogram(data, x="Gender", color="Gender", color_discrete_map=color_map)
            st.title('Histogram')
            st.plotly_chart(fig)
            st.title('Pie Chart')
            figpie = px.pie(data, names="Gender", color="Gender", color_discrete_map=color_map)
            st.plotly_chart(figpie)

        elif viz_type == "Blood Type":
            column = "Blood Type" 
            color_map = {"A": "red", "B": "green", "AB": "purple", "O": "blue"}  # Define the color map for Blood Type
            fig = px.histogram(data, x=column, color=column, color_discrete_map=color_map)
            st.title('Histogram')
            st.plotly_chart(fig)
            st.title('Pie Chart')
            figpie = px.pie(data, names=column, color=column, color_discrete_map=color_map)
            st.plotly_chart(figpie)

        elif viz_type == "Medical Condition":
            column = "Medical Condition"  
            color_map = {"Condition 1": "cyan", "Condition 2": "magenta", "Condition 3": "yellow", "Condition 4": "green"}  # Define the color map for Medical Condition
            fig = px.histogram(data, x=column, color=column, color_discrete_map=color_map)
            st.title('Histogram')
            st.plotly_chart(fig)
            st.title('Pie Chart')
            figpie = px.pie(data, names=column, color=column, color_discrete_map=color_map)
            st.plotly_chart(figpie)

        elif viz_type == "Insurance Provider":
            column = "Insurance Provider" 
            color_map = {"Provider 1": "orange", "Provider 2": "purple", "Provider 3": "green", "Provider 4": "blue"}  # Define the color map for Insurance Provider
            fig = px.histogram(data, x=column, color=column, color_discrete_map=color_map)
            st.title('Histogram')
            st.plotly_chart(fig)
            st.title('Pie Chart')
            figpie = px.pie(data, names=column, color=column, color_discrete_map=color_map)
            st.plotly_chart(figpie)

        elif viz_type == "Admission Type":
            column = "Admission Type"  
            color_map = {"Elective": "Green", "Emergency": "Red", "Urgent": "Orange"}  # Define the color map for Admission Type
            fig = px.histogram(data, x=column, color=column, color_discrete_map=color_map)
            st.title('Histogram')
            st.plotly_chart(fig)
            st.title('Pie Chart')
            figpie = px.pie(data, names=column, color=column, color_discrete_map=color_map)
            st.plotly_chart(figpie)

        elif viz_type == "Medication":
            column = "Medication" 
            color_map = {"Medication A": "orange", "Medication B": "purple", "Medication C": "green", "Medication D": "blue"}  # Define the color map for Medication
            fig = px.histogram(data, x=column, color=column, color_discrete_map=color_map)
            st.title('Histogram')
            st.plotly_chart(fig)
            st.title('Pie Chart')
            figpie = px.pie(data, names=column, color=column, color_discrete_map=color_map)
            st.plotly_chart(figpie)

        elif viz_type == "Test Results":
            column = "Test Results"
            color_map = {"Abnormal": "Red", "Inconclusive": "yellow", "Normal": "green"}  # Define the color map for Test Results
            fig = px.histogram(data, x=column, color=column, color_discrete_map=color_map)
            st.title('Histogram')
            st.plotly_chart(fig)
            st.title('Pie Chart')
            figpie = px.pie(data, names=column, color=column, color_discrete_map=color_map)
            st.plotly_chart(figpie)

    elif page == "Comparision Visualizations":
        st.subheader("Pick Comparisons")
        compare_type = st.selectbox("Select Comparison Based On", ["Billing analysis for Each Field", "Billing Amount according to Medical Condition and Medication", "Billing Amount according to Medical Condition and Test Results", "Highest Features according to average number of days hospitalized"])    
        if compare_type == "Highest Features according to Billing Amount":
            viz_type = st.selectbox("Select Visualization Based On", ["Highest Gender according to Billing Amount", "Highest Blood Type according to Billing Amount", "Highest Insurance Provider according to Billing Amount", "Highest Test Results according to Billing Amount", "Highest Medication according to Billing Amount", "Highest Admission Type according to Billing Amount","Highest Hospital according to Billing Amount","Highest Doctor according to Billing Amount","Highest Medical Condition according to Billing Amount"])    
            if viz_type == "Highest Gender according to Billing Amount":
                for i in cols:
                    if i == 'Gender':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"], marker_color=["pink", "blue"]))
                        fig.update_layout(title="Highest Gender According to " + 'Billing Amount',
                          xaxis_title='Gender',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig) 
            elif viz_type == "Highest Blood Type according to Billing Amount":
                for i in cols:
                    if i == 'Blood Type':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        color_map = {"A+": "red", "B+": "green", "AB-": "purple", "O+": "blue", "AB+": "yellow", "O-": "orange", "A-": "white", "B-": "brown"}
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"], marker_color=[color_map[blood_type] for blood_type in chart_data[i]]))
                        fig.update_layout(title="Highest Blood Type According to " + 'Billing Amount',
                          xaxis_title='Blood Type',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig)              
            elif viz_type == "Highest Insurance Provider according to Billing Amount":
                for i in cols:
                    if i == 'Insurance Provider':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"], marker_color=["blue", "yellow", "green", "brown", "orange"]))
                        fig.update_layout(title="Highest Insurance Provider According to " + 'Billing Amount',
                          xaxis_title='Insurance Provider',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig) 
            elif viz_type == "Highest Test Results according to Billing Amount":
                for i in cols:
                    if i == 'Test Results':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"], marker_color=["red", "orange", "green"]))
                        fig.update_layout(title="Highest Test Results According to " + 'Billing Amount',
                          xaxis_title='Test Results',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig) 
            elif viz_type == "Highest Medication according to Billing Amount":
                for i in cols:
                    if i == 'Medication':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"], marker_color=["blue", "yellow", "green", "brown", "orange"]))
                        fig.update_layout(title="Highest Medication according to Billing Amount " + 'Billing Amount',
                          xaxis_title='Highest Medication according to Billing Amount',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig) 
            elif viz_type == "Highest Admission Type according to Billing Amount":
                for i in cols:
                    if i == 'Admission Type':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"], marker_color=["orange", "green", "red"]))
                        fig.update_layout(title="Highest Admission Type According to " + 'Billing Amount',
                          xaxis_title='Admission Type',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig) 
            elif viz_type == "Highest Hospital according to Billing Amount":
                for i in cols:
                    if i == 'Hospital':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"]))
                        fig.update_layout(title="Highest Hospital according to " + 'Billing Amount',
                          xaxis_title='Hospital',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig)
            elif viz_type == "Highest Doctor according to Billing Amount":
                for i in cols:
                    if i == 'Doctor':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"]))
                        fig.update_layout(title="Highest Doctor according to Billing Amount " + 'Billing Amount',
                          xaxis_title='Highest Doctor according to Billing Amount',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig) 
            elif viz_type == "Highest Medical Condition according to Billing Amount":
                for i in cols:
                    if i == 'Medical Condition':
                        chart_data = data.groupby([i])[['Billing Amount']].sum().reset_index()
                        chart_data = chart_data.sort_values(by=("Billing Amount"), ascending=False)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=chart_data[i], y=chart_data["Billing Amount"], marker_color=["blue", "yellow", "green", "brown", "orange", "white"]))
                        fig.update_layout(title="Highest Medical Condition According to " + 'Billing Amount',
                          xaxis_title='Medical Condition',
                          yaxis_title= "Billing Amount",
                          plot_bgcolor='black', 
                          paper_bgcolor='black',  
                          font=dict(color='black'))
                        st.plotly_chart(fig)  
        elif compare_type == "Billing Amount according to Medical Condition and Medication":
            df_trans = data.groupby(['Medical Condition', 'Medication'])[['Billing Amount']].sum().reset_index()
            plt.figure(figsize=(15, 6))
            sns.barplot(x=df_trans['Medical Condition'], y=df_trans['Billing Amount'], hue=df_trans['Medication'], ci=None, palette="Set1")
            plt.title("Billing Amount according to Medical Condition and Medication")
            plt.ylabel("Billing Amount")
            plt.xticks(rotation=45, fontsize=9)
            st.pyplot(plt)  
        elif compare_type == "Billing Amount according to Medical Condition and Test Results":
            df_trans = data.groupby(['Medical Condition', 'Test Results'])[['Billing Amount']].sum().reset_index()
            plt.figure(figsize=(15, 6))
            sns.barplot(x=df_trans['Medical Condition'], y=df_trans['Billing Amount'], hue=df_trans['Test Results'], ci=None, palette="Set1")
            plt.title("Billing Amount according to Medical Condition and Test Results")
            plt.ylabel("Billing Amount")
            plt.xticks(rotation=45, fontsize=9)
            st.pyplot(plt) 
        elif compare_type == "Highest Features according to average number of days hospitalized":
            viz_type = st.selectbox("Select Visualization Based On", ["Highest Gender according to average number of days hospitalized", "Highest Blood Type according to average number of days hospitalized", "Highest Insurance Provider according average number of days hospitalized", "Highest Test Results according to average number of days hospitalized", "Highest Medication according to average number of days hospitalized", "Highest Admission Type according to average number of days hospitalized","Highest Hospital according to average number of days hospitalized","Highest Doctor according to average number of days hospitalized","Highest Medical Condition according to average number of days hospitalized"])    
            if viz_type == "Highest Gender according to average number of days hospitalized":
                for i in cols:
                    if i == 'Gender':
                        char_bar = data.groupby(['Gender'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Gender'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Gender according to average number of days hospitalized',
                          xaxis_title='Gender',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig)
            elif viz_type == "Highest Hospital according to average number of days hospitalized":
                for i in cols:
                    if i == 'Hospital':
                        char_bar = data.groupby(['Hospital'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Hospital'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Hospital according to average number of days hospitalized',
                          xaxis_title='Hospital',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig)
            elif viz_type == "Highest Insurance Provider according to average number of days hospitalized":
                for i in cols:
                    if i == 'Insurance Provider':
                        char_bar = data.groupby(['Insurance Provider'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Insurance Provider'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Insurance Provider according to average number of days hospitalized',
                          xaxis_title='Insurance Provider',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig)
            elif viz_type == "Highest Admission Type according to average number of days hospitalized":
                for i in cols:
                    if i == 'Admission Type':
                        char_bar = data.groupby(['Admission Type'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Admission Type'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Admission Type according to average number of days hospitalized',
                          xaxis_title='Admission Type',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig)                        
            elif viz_type == "Highest Doctor according to average number of days hospitalized":
                for i in cols:
                    if i == 'Doctor':
                        char_bar = data.groupby(['Doctor'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Doctor'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Doctor according to average number of days hospitalized',
                          xaxis_title='Doctor',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig)            
            elif viz_type == "Highest Blood Type according to average number of days hospitalized":
                for i in cols:
                    if i == 'Blood Type':
                        char_bar = data.groupby(['Blood Type'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Blood Type'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Blood Type according to average number of days hospitalized',
                          xaxis_title='Blood Type',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig)
            elif viz_type == "Highest Medical Condition according to average number of days hospitalized":
                for i in cols:
                    if i == 'Medical Condition':
                        char_bar = data.groupby(['Medical Condition'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Medical Condition'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Medical Condition according to average number of days hospitalized',
                          xaxis_title='Medical Condition',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig) 
            elif viz_type == "Highest Test Results according to average number of days hospitalized":
                for i in cols:
                    if i == 'Test Results':
                        char_bar = data.groupby(['Test Results'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Test Results'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Test Results according to average number of days hospitalized',
                          xaxis_title='Test Results',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig)   
            elif viz_type == "Highest Medication according to average number of days hospitalized":
                for i in cols:
                    if i == 'Medication':
                        char_bar = data.groupby(['Medical Condition'])[['Days hospitalized']].mean().reset_index()
                        char_bar = char_bar.sort_values(by=("Days hospitalized"), ascending=False)
                        top = char_bar.head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=top['Medical Condition'], y=top["Days hospitalized"]))
                        fig.update_layout(title='Highest Medical Condition according to average number of days hospitalized',
                          xaxis_title='Medical Condition',
                          yaxis_title="Days hospitalized",
                          plot_bgcolor='black', 
                          paper_bgcolor='gray',  
                          font=dict(color='white'))
                        st.plotly_chart(fig) 
    elif page == "Trends Visualizations":
        st.subheader("Trends Visualizations")
        viz_type = st.selectbox("Select Visualization Based On", ["Hospital Admission Trends", "Medications with test results", "Age vs Billing Amount", "Admissions Over Time","Top 10 Doctors"])                                                          
        if viz_type == "Hospital Admission Trends":
            admission_type = st.sidebar.multiselect(
            "Select Admission Type",
            options=data['Admission Type'].unique(),
            default=data['Admission Type'].unique()
            )
            filtered_data = data[data['Admission Type'].isin(admission_type)]
            grouped_data = filtered_data.groupby(['Date of Admission', 'Admission Type']).size().reset_index(name='counts')
            fig = px.line(grouped_data, x="Date of Admission", y="counts", color='Admission Type', 
              title="Admission Trends Over Time", labels={'counts': 'Number of Admissions'})

            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Medications with test results":
            selected_condition = st.sidebar.selectbox(
            "Select Medical Condition",
            options=["All"] + list(data['Medical Condition'].unique())
            )

            selected_age_group = st.sidebar.slider(
            "Select Age Range",
            min_value=int(data['Age'].min()), 
            max_value=int(data['Age'].max()), 
            value=(int(data['Age'].min()), int(data['Age'].max()))
            )
            if selected_condition != "All":
                data = data[data['Medical Condition'] == selected_condition]

                data = data[(data['Age'] >= selected_age_group[0]) & (data['Age'] <= selected_age_group[1])]
                fig = px.scatter(data, x="Medication", y="Test Results", size="Age", color="Medical Condition",
                 title="Medication vs. Test Results", hover_data=['Name', 'Age'])

                st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Admissions Over Time":
            st.sidebar.header('Filters')
            year = st.sidebar.multiselect('Select Year', options=data['Admission Year'].unique(), default=data['Admission Year'].unique())
            admission_type = st.sidebar.multiselect('Select Admission Type', options=data['Admission Type'].unique(), default=data['Admission Type'].unique())
            filtered_data = data
            if year:
                filtered_data = filtered_data[filtered_data['Admission Year'].isin(year)]
            if admission_type:
                filtered_data = filtered_data[filtered_data['Admission Type'].isin(admission_type)]
            st.header('Admission Types Over Time')
            chart_type = st.selectbox("Select Chart Type", ["Line", "Bar"])
            admission_types_over_time = filtered_data.groupby(['Admission Year', 'Admission Type']).size().unstack()
            fig, ax = plt.subplots()
            if chart_type == "Line":
                admission_types_over_time.plot(kind='line', marker='o', ax=ax)
            else:
                admission_types_over_time.plot(kind='bar', stacked=True, ax=ax)
            plt.title('Admission Types Over Time')
            plt.xlabel('Year')
            plt.ylabel('Count')
            st.pyplot(fig)
        elif viz_type == "Age vs Billing Amount":
            st.sidebar.header('Filters')
            age_min, age_max = st.sidebar.slider('Select Age Range', int(data['Age'].min()), int(data['Age'].max()), (int(data['Age'].min()), int(data['Age'].max())))
            billing_min, billing_max = st.sidebar.slider('Select Billing Range', float(data['Billing Amount'].min()), float(data['Billing Amount'].max()), (float(data['Billing Amount'].min()), float(data['Billing Amount'].max())))
            filtered_data = data[(data['Age'] >= age_min) & (data['Age'] <= age_max) & (data['Billing Amount'] >= billing_min) & (data['Billing Amount'] <= billing_max)]
            st.header('Age vs Billing Amount')
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_data, x='Age', y='Billing Amount', ax=ax)
            plt.title('Age vs Billing Amount')
            plt.xlabel('Age')
            plt.ylabel('Billing Amount')
            st.pyplot(fig)
        elif viz_type == "Top 10 Doctors":
            st.header('Top Doctors by Patient Load')
            num_doctors = st.slider('Select Number of Top Doctors', 1, 20, 10)
            doctors_patient_load = data['Doctor'].value_counts().head(num_doctors)
            fig, ax = plt.subplots()
            doctors_patient_load.plot(kind='bar', ax=ax)
            plt.title(f'Top {num_doctors} Doctors by Patient Load')
            plt.xlabel('Doctor')
            plt.ylabel('Number of Patients')
            st.pyplot(fig)
    elif page == "Machine Learning":
        try:
    # Preprocessing steps
            healthcare_data = pd.read_csv('healthcare_dataset.csv')
            healthcare_data.head()
            le = LabelEncoder()
            healthcare_data['Medical Condition'] = le.fit_transform(healthcare_data['Medical Condition'])
            healthcare_data['Gender'] = healthcare_data['Gender'].map({'Male': 0, 'Female': 1})
            X = healthcare_data[['Age', 'Gender', 'Medical Condition']]
            y = healthcare_data['Billing Amount']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree Regressor
            dt_regressor = DecisionTreeRegressor(random_state=42)
            dt_regressor.fit(X_train, y_train)
            y_pred_dt = dt_regressor.predict(X_test)
            mse_dt = mean_squared_error(y_test, y_pred_dt)

    # Bayesian Ridge Regressor
            br_regressor = BayesianRidge()
            br_regressor.fit(X_train, y_train)
            y_pred_br = br_regressor.predict(X_test)
            mse_br = mean_squared_error(y_test, y_pred_br)

    # Display the MSE values on Streamlit
            st.write(f"Decision Tree MSE: {mse_dt}")
            st.write(f"Bayesian Ridge MSE: {mse_br}")

        except Exception as e:
            st.write(f"An error occurred: {e}")
    elif page == "Machine Learning Model Training & Prediction":
        healthcare_data = pd.read_csv('healthcare_dataset.csv')
        healthcare_data.head()
        data = pd.read_csv('healthcare_dataset.csv')
        viz_type = st.selectbox("Select Prediction Based On", ["MLClassifier", "All Models Results","Label Spreading","For Single Patient"]) 

        # Select relevant features and target variable
        X = data[['Age', 'Gender', 'Medical Condition', 'Admission Type']]
        y = data['Test Results']  

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize numerical features
        scaler = StandardScaler()
        X_train[['Age']] = scaler.fit_transform(X_train[['Age']])
        X_test[['Age']] = scaler.transform(X_test[['Age']])

        # Define the transformers for numerical and categorical columns
        numerical_features = ['Age']
        categorical_features = ['Gender', 'Medical Condition', 'Admission Type']

        # Create a preprocessing pipeline
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
        ])

        # Create the full pipeline with the classifier
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', MLPClassifier(max_iter=250, activation="logistic"))])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # LazyClassifier
        clf = LazyClassifier(predictions=True)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)

        # Streamlit App
        st.title('Healthcare Outcome Prediction')

        # Model Evaluation Section
        y_pred_risk = pipeline.predict(X_test)
        if viz_type == 'MLClassifier':
            accuracy = accuracy_score(y_test, y_pred_risk)
            report = classification_report(y_test, y_pred_risk)
            st.header('Model Evaluation')
            st.subheader('Classification Report')
            st.text(f"Accuracy: {accuracy_score(y_test, y_pred_risk)}")
            st.text("Classification Report:")
            st.text(report)
            
        elif viz_type == 'Label Spreading':
            accuracy = accuracy_score(y_test, y_pred_risk)
            report = classification_report(y_test, y_pred_risk)
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
            preprocessor = ColumnTransformer(
            transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
            ])
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LabelSpreading(max_iter=30))])
            st.header('Model Evaluation')
            st.subheader('Classification Report')
            st.text(f"Accuracy: {accuracy_score(y_test, y_pred_risk)}")
            st.text("Classification Report:")
            st.text(report)
        # Fit the model
            pipeline.fit(X_train, y_train)
        elif viz_type == 'All Models Results':
            st.header('All Classifier Model Results')
            st.text('Classifier Results:')
            st.dataframe(models)
        elif viz_type == 'For Single Patient': 
            # Sidebar
            st.sidebar.header('Enter Patient Information')

            age = st.sidebar.slider('Age', 0, 100, 35)
            gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
            medical_condition = st.sidebar.selectbox('Medical Condition', ['Asthma', 'Obesity','Diabetes','Arthritis','Cancer','Hypertension'])
            admission_type = st.sidebar.selectbox('Admission Type', ['Emergency', 'Elective','Urgent'])

            single_patient_data = {
                'Age': age,
                'Gender': gender,
                'Medical Condition': medical_condition,
                'Admission Type': admission_type
            }

            # Make prediction
            if st.sidebar.button('Predict Outcome'):
            # Create a DataFrame for the single row
                single_patient_df = pd.DataFrame([single_patient_data])

    # Standardize the 'Age' feature using the previously fitted StandardScaler
                single_patient_df[['Age']] = scaler.transform(single_patient_df[['Age']])

    # Use the trained model to make a prediction
                outcome_prediction = pipeline.predict(single_patient_df)

    # Display the predicted outcome
                st.sidebar.subheader('Predicted Outcome:')
                st.sidebar.write(outcome_prediction[0])       
# Run the app
if __name__ == "__main__":
    main()
