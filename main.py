from dash import Dash,html,dcc,Input,Output
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

#model
from sklearn import linear_model,tree,neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

models = [XGBClassifier(),DecisionTreeRegressor(),
           LogisticRegression(max_iter=2000) 
           ,RandomForestClassifier(), SVC(), 
             KNeighborsClassifier()]

# Define the calculate_metrics function outside the callback functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error

def calculate_metrics(y_test, y_pred, metric_type):
    if metric_type == 'Accuracy':
        return accuracy_score(y_test, y_pred)
    elif metric_type == 'Precision':
        return precision_score(y_test, y_pred)
    elif metric_type == 'Recall':
        return recall_score(y_test, y_pred)
    elif metric_type == 'MSE':
        return mean_squared_error(y_test, y_pred)
    elif metric_type == 'MAE':
        return mean_absolute_error(y_test, y_pred)
    # You can add more error metrics here
    return 0


# Create an empty Div to display accuracy
accuracy_div = html.Div(id='accuracy-div', children=[])

# Dropdown for error metrics
error_metric_dropdown = dcc.Dropdown(
    id='error-metric-dropdown',
    options=[
        {'label': 'Accuracy', 'value': 'Accuracy'},
        {'label': 'Precision', 'value': 'Precision'},
        {'label': 'Recall', 'value': 'Recall'},
        {'label': 'Mean Square Error', 'value': 'MSE'},
        {'label': 'Mean Absolute Error', 'value': 'MAE'}
        # Add more error metrics as needed
    ],
    value='Accuracy'  # Default selected metric
)


#Layout
app.layout = html.Div([
    html.H4("Breast Cancer Prediction",style={'textAlign': 'center'}),
    html.P("Select Model"),
    dcc.Dropdown(
        id='dropdown_id',
        options=['Regression','Decision Tree','RandomForestClassifier','k-NN'],
        value='Regression',
        clearable=False
    ),
    
    dcc.Graph(id='graph_id'),
    # Add the accuracy div
    accuracy_div,
    # Add the error metric dropdown
    html.H4('Error Metric'),
    error_metric_dropdown,
    # Add the error metric div
    html.Div(id='error-metric-div')
    

])

@app.callback(
    Output('graph_id', 'figure'),
    Output('accuracy-div', 'children'),
    Output('error-metric-div', 'children'),  # Output for error metric display
    Input('dropdown_id', 'value'),
    Input('error-metric-dropdown', 'value')  # Input for error metric selection
)

#function for training model
def train_and_testing(model_name,error_metric_type):
    df = pd.read_csv('breast-cancer.csv')
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    X = df.drop('diagnosis',axis=1)
    y = df.diagnosis
    X_train, X_test,y_train,y_test =train_test_split(
        X,y,test_size=0.2,random_state=42)
    accuracy_values = []  # Define accuracy_values here
    for model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        model_name = type(model).__name__
        #print(f'{model_name}- Accuracy:{accuracy:.2f}')
        #accuracy_values.append(f'{model_name} - Accuracy: {accuracy:.2f}\n')
    # Calculate the selected error metric
    error_metric_text = calculate_metrics(y_test, y_pred, error_metric_type)

    # Scatter Plot
    scatter_fig = go.Figure(data=go.Scatter(
        x=df['radius_mean'],
        y=df['texture_mean'],
        mode='markers',
        marker=dict(color=df['diagnosis'], colorscale='Viridis', showscale=True),
        text=df['diagnosis']
    ))
    scatter_fig.update_layout(title='Scatter Plot', xaxis_title='Radius Mean', yaxis_title='Texture Mean')

    accuracy_text = "\n".join(accuracy_values)  # Combine accuracy values with new lines

    return scatter_fig, accuracy_text,error_metric_text # Return accuracy values as text
    #return fig,accuracy_values


app.run_server(debug=True)

