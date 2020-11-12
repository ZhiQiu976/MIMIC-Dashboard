import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

from app import app

# function1
def plot_confusion_matrix(cm, labels):
    '''
    Function for plotting confusion matrix (normalized).
    
    cm : confusion matrix list(list)
    labels : name of the data list(str)
    title : title for the heatmap
    '''
    
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "white"},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations,
        "margin": dict(t=0)
    }
    fig = go.Figure(data=data, layout=layout)
    return fig

# function2
def plot_df(df_test, df_train):
    
    colors = ['rgb(189, 215, 231)', 'rgb(107, 174, 214)',
          'rgb(49, 130, 189)', 'rgb(8, 81, 156)']
    
    df = pd.DataFrame(columns=['Training', 'Testing'], index=['Accuracy', 'F1(weighted)',
                                                          'Recall(weighted)', 'Precision(weighted)'])

    for i, df_f in enumerate([df_train, df_test]):
        accuracy = accuracy_score(df_f.iloc[:,0], df_f.iloc[:,1])
        f1 = f1_score(df_f.iloc[:,0], df_f.iloc[:,1], average='weighted')
        recall = recall_score(df_f.iloc[:,0], df_f.iloc[:,1], average='weighted')
        precision = precision_score(df_f.iloc[:,0], df_f.iloc[:,1], average='weighted', zero_division=0)
        result = np.array([accuracy, f1, recall, precision]).round(2)
        df.iloc[:,i] = result
        
    df = df.reset_index().rename(columns={'index':'Metric'})
    df['Color'] = colors
    
    layout = {
        "margin": dict(t=0)
    }
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(['Metric', 'Training', 'Testing']),
                    line_color='white', fill_color='white',
                    align='left', font=dict(color='black', size=12)),
        cells=dict(values=[df.Metric, df.Training, df.Testing],
                   line_color=[df.Color], fill_color=[df.Color],
                    align='left', font=dict(color='black', size=11))
    )
    ])
    
    return fig



# data
df_train = pd.read_csv('./Data/train_results_decoded.csv', index_col=0)
df_test = pd.read_csv('./Data/test_results_decoded.csv', index_col=0)

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1(children='Model Performance and Predictions for Discharge Locations'), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Visualization of model performance, feature importance and prediction results.'), className="mb-4")
        ]),
# choose between classifiers
    dcc.Dropdown(
        id='classifier',
        options=[
            {'label': 'Dummy Classifier', 'value': 'dummy'}, 
            {'label': 'kNN', 'value': 'knn'}, 
            {'label': 'Logistic Regression', 'value': 'logreg'},
            {'label': 'SVM', 'value': 'svm'},
            {'label': 'XgBoost', 'value': 'xgb'},
        ],
        value='xgb',
        #multi=True,
        style={'width': '50%'}
        ),
# for some reason, font colour remains black if using the color option
    dbc.Row([
        dbc.Col(dbc.Card(html.H3(children='Model Performance Evaluation',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
    ]),
    dbc.Row([
        dbc.Col(html.H5(children='Confusion Matrix - Testing', className="text-center"), width=7, className="mt-4"),
        dbc.Col(html.H5(children='Evaluation Metrics', className="text-center"), width=5, className="mt-4")
        ]),


    dbc.Row([
        dbc.Col(dcc.Graph(id='conf1'), width=7), 
        dbc.Col(dcc.Graph(id='table1'), width=5) 
        ]),

])


])

@app.callback([Output('conf1', 'figure'),
               Output('table1', 'figure')],
              [Input('classifier', 'value')])

def update_graph(classifier):
    
    target_label = ['DEAD/EXPIRED', 'HOME', 'OTHERS', 'SNF']
    
    # different model
    if classifier=="dummy":
        df_fig = df_test.copy().iloc[:, [0,1]]
        df_fig2 = df_train.copy().iloc[:, :2]
        cm = confusion_matrix(df_fig.iloc[:,0], df_fig.iloc[:,1])
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)).round(2)
        
    elif classifier=="knn":
        df_fig = df_test.copy().iloc[:, [0,2]]
        df_fig2 = df_train.copy().iloc[:, [0,2]]
        cm = confusion_matrix(df_fig.iloc[:,0], df_fig.iloc[:,1])
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)).round(2)
        
    elif classifier=="logreg":
        df_fig = df_test.copy().iloc[:, [0,3]]
        df_fig2 = df_train.copy().iloc[:, [0,3]]
        cm = confusion_matrix(df_fig.iloc[:,0], df_fig.iloc[:,1])
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)).round(2)
        
    elif classifier=="svm":
        df_fig = df_test.copy().iloc[:, [0,4]]
        df_fig2 = df_train.copy().iloc[:, [0,4]]
        cm = confusion_matrix(df_fig.iloc[:,0], df_fig.iloc[:,1])
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)).round(2)
        
    else:
        df_fig = df_test.copy().iloc[:, [0,5]]
        df_fig2 = df_train.copy().iloc[:, [0,5]]
        cm = confusion_matrix(df_fig.iloc[:,0], df_fig.iloc[:,1])
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)).round(2)
        

    # figure 1, confusion matrix
    fig = plot_confusion_matrix(cm_percent, target_label)
    
    # figure 2, table
    fig2 = plot_df(df_fig, df_fig2)

    return fig, fig2

# needed only if running this as a single page app
#if __name__ == '__main__':
#    app.run_server(host='127.0.0.1', debug=True)