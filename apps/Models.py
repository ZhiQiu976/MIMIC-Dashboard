import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
    '''
    Function for plotting dataframe as table.
    
    df_test : testing data (dataframe)
    df_train : training data (dataframe)
    '''
    
    colors = ['rgb(189, 215, 231)', 'rgb(107, 174, 214)',
          'rgb(49, 130, 189)', 'rgb(8, 81, 156)']
    
    df = pd.DataFrame(columns=['Training', 'Testing'], index=['Accuracy', 'F1',
                                                          'Recall', 'Precision'])

    for i, df_f in enumerate([df_train, df_test]):
        accuracy = accuracy_score(df_f.iloc[:,0], df_f.iloc[:,1])
        f1 = f1_score(df_f.iloc[:,0], df_f.iloc[:,1], average='weighted')
        recall = recall_score(df_f.iloc[:,0], df_f.iloc[:,1], average='weighted')
        precision = precision_score(df_f.iloc[:,0], df_f.iloc[:,1], average='weighted', zero_division=0)
        result = np.array([accuracy, f1, recall, precision]).round(2)
        df.iloc[:,i] = result
        
    df = df.reset_index().rename(columns={'index':'Metric'})
#     df['Color'] = colors
    
#     layout = {
#         "margin": dict(t=0)
#     }
    
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=list(['Metric', 'Training', 'Testing']),
#                     line_color='white', fill_color='white',
#                     align='left', font=dict(color='black', size=12)),
#         cells=dict(values=[df.Metric, df.Training, df.Testing],
#                    line_color=[df.Color], fill_color=[df.Color],
#                     align='left', font=dict(color='black', size=11))
#     )
#     ])
    
    fig = ff.create_table(df)
    
    return fig

# helper function for graph layput
def size_adjustment(n):
    '''
    Helper function for controlling graph size.
    
    n : number of input to be shown later (int)
    '''
    
    if n is None:
        best = dict(height=2000, width=1000)
    else:
        s = n-1
        if s in range(6):
            best = dict(height=350, width=1000)
        elif s in range(6, 11):
            best = dict(height=500, width=1000)
        elif s in range(11, 19):
            best = dict(height=700, width=1000)
        elif s in range(19, 36):
            best = dict(height=1000, width=1000)
        elif s in range(36, 44):
            best = dict(height=1200, width=1000)
        elif s in range(44, 51):
            best = dict(height=1500, width=1000)
        elif s in range(51, 61):
            best = dict(height=1700, width=1000)
        elif s in range(61, 71):
            best = dict(height=1850, width=1000)
        else:
            best = dict(height=2000, width=1000)
            
    return best

# function 3
def plot_imp_single(df, n):
    '''
    Function for plot feature importance for a single classifier.
    
    df: input feature importance dataframe (dataframe)
    n : number of features to display (float)
    '''
    
    df_new = df.sort_values(by = ['importance'], ascending=False)[:n]
    df_new_ = df_new.sort_values(by = ['importance'])
    
    best = size_adjustment(n)
    
    layout = dict(
        title='Barplot of Feature Importance with top {} features'.format(n),
        **best,
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            automargin=True
        ))


    fig = go.Figure(go.Bar(
            x=[i for i in df_new_.importance],
            y=[i for i in df_new_.features],
            marker={
            "color": [i for i in df_new_.importance], 
            "colorscale": "Viridis", 
            "reversescale": True
          }, 
            orientation='h'),
                   layout=layout)

    return fig

# function 4
def plot_multi(df, n):
    '''
    Function for plot feature importance for different classes in a single classifier.
    Final ranking is computed by summing up absolute values of coefficient for all classes.
    
    df: input feature importance dataframe (dataframe)
    n : number of features to display (float)
    '''
    
    best = size_adjustment(n)
    
    df['sum_importance'] = abs(df.home)+abs(df.snf)+abs(df.others)+abs(df.dead)
    df_new = df.sort_values(by = ['sum_importance'], ascending=False)[:n]
    df_new_ = df_new.sort_values(by = ['sum_importance'])

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Home',
                         y=[i for i in df_new_.features],
                         x=[i for i in df_new_.home], orientation='h'))
    fig.add_trace(go.Bar(name='SNF', 
                         y=[i for i in df_new_.features], 
                         x=[i for i in df_new_.snf], orientation='h'))
    fig.add_trace(go.Bar(name='Others', 
                         y=[i for i in df_new_.features],
                         x=[i for i in df_new_.others], orientation='h'))
    fig.add_trace(go.Bar(name='Dead/Expired', 
                         y=[i for i in df_new_.features], 
                         x=[i for i in df_new_.dead], orientation='h'))

    fig.update_layout(barmode='relative', 
                      title='Barplot of Feature Importance with top {} features (ranked by sum of absolute values)'.format(n), 
                      **best)

    return fig


# data
df_train = pd.read_csv('./Data/train_results_decoded.csv', index_col=0)
df_test = pd.read_csv('./Data/test_results_decoded.csv', index_col=0)


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1(children='Model Evaluation'), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Visualization of model performance and feature importance for Target Variable - Discharge Locations.'), className="mb-4")
        ]),
        
        
    # for some reason, font colour remains black if using the color option
    dbc.Row([
        dbc.Col(dbc.Card(html.H3(children='Model Performance',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
    ]),
        
        
    # choose between classifiers
    html.Label(["Select a model: "]),        
     dcc.Dropdown(
        id='classifier',
        options=[
            {'label': 'Dummy Classifier', 'value': 'dummy'}, 
            {'label': 'kNN', 'value': 'knn'}, 
            {'label': 'Logistic Regression', 'value': 'logreg'},
            {'label': 'SVM', 'value': 'svm'},
            {'label': 'XgBoost', 'value': 'xgb'},
            {'label': 'Random Forest', 'value': 'rf'}
        ],
        value='xgb',
        #multi=True,
        #style={'width': '50%'}
        ),    
   
        
    dbc.Row([
        dbc.Col(html.H5(children='Confusion Matrix - Testing', className="text-center"), width=7, className="mt-4"),
        dbc.Col(html.H5(children='Evaluation Metrics', className="text-center"), width=5, className="mt-4")
        ]),


    dbc.Row([
        dbc.Col(dcc.Graph(id='conf1'), width=7), 
        dbc.Col(dcc.Graph(id='table1'), width=5) 
        ]),
    
    # feature importance part
    dbc.Row([
        dbc.Col(dbc.Card(html.H3(children='Feature Importance Exploration',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
    ]),    
    
        
#     dcc.Input(
#             id="num_of_feature", type="number", placeholder='Select or enter...',
#             min=1, max=84,
#         ),
     dbc.FormGroup(
        [
            dbc.Label("Classifiers with built-in importance ranking:"),
            dbc.RadioItems(
                id='classifier2',
                options=[
                    {'label': 'Logistic Regression', 'value': 'logreg'},
                    {'label': 'XgBoost', 'value': 'xgb'},
                    {'label': 'Random Forest', 'value': 'rf'},
                ],
                value='xgb',
                inline=True
            ),
        ]
    ),
        
    html.Br(),
        
    html.Label(["Number of Top features: (minimum 1, maximum 84)"]),
    html.Br(),
        
    dcc.Slider(
        id="num_of_feature",
        min=1,
        max=84,
        #marks={i: '{}'.format(i) for i in range(84)},
        value=10,
    ),  
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='imp1'), width=12)
        ]),
    

])


])

@app.callback([Output('conf1', 'figure'),
               Output('table1', 'figure')],
              [Input('classifier', 'value')])

def update_graph1(classifier):
    
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
        
    elif classifier=='xgb':
        df_fig = df_test.copy().iloc[:, [0,5]]
        df_fig2 = df_train.copy().iloc[:, [0,5]]
        cm = confusion_matrix(df_fig.iloc[:,0], df_fig.iloc[:,1])
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)).round(2)
        df_imp = pd.read_csv('./Data/xgb_importance.csv', index_col=0)
        
    else:
        df_fig = df_test.copy().iloc[:, [0,6]]
        df_fig2 = df_train.copy().iloc[:, [0,6]]
        cm = confusion_matrix(df_fig.iloc[:,0], df_fig.iloc[:,1])
        cm_percent = (cm / cm.sum(axis=1, keepdims=True)).round(2)
        df_imp = pd.read_csv('./Data/rf_importance.csv', index_col=0)
        
      
    # figure 1, confusion matrix
    fig = plot_confusion_matrix(cm_percent, target_label)
    
    # figure 2, table
    fig2 = plot_df(df_fig, df_fig2)

    return fig, fig2



@app.callback(Output('imp1', 'figure'),
              [Input('classifier2', 'value'), Input('num_of_feature', 'value')])

def update_graph2(classifier2, num_of_feature):
    
    target_label = ['DEAD/EXPIRED', 'HOME', 'OTHERS', 'SNF']
    
    # different model
    if classifier2=="logreg":
        df_imp = pd.read_csv('./Data/logreg_importance.csv', index_col=0)
        fig3 = plot_multi(df_imp, num_of_feature)
        
    elif classifier2=='xgb':
        df_imp = pd.read_csv('./Data/xgb_importance.csv', index_col=0)
        fig3 = plot_imp_single(df_imp, num_of_feature)
        
    elif classifier2=='rf':
        df_imp = pd.read_csv('./Data/rf_importance.csv', index_col=0)
        fig3 = plot_imp_single(df_imp, num_of_feature)

    return fig3




# needed only if running this as a single page app
#if __name__ == '__main__':
#    app.run_server(host='127.0.0.1', debug=True)
