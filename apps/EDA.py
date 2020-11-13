import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd

from app import app

# needed if running single page dash app instead
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

y_train = pd.read_csv("./Data/y_train.csv",index_col=0)
y_test = pd.read_csv("./Data/y_test.csv",index_col=0)

X_train = pd.read_csv("./Data/X_train.csv",index_col=0)
X_test = pd.read_csv("./Data/X_test.csv",index_col=0)

train_time = pd.read_csv("./Data/train_time.csv", index_col=0)
test_time = pd.read_csv("./Data/test_time.csv", index_col=0)

train_year = pd.to_datetime(train_time.DISCHTIME).dt.year
test_year = pd.to_datetime(test_time.DISCHTIME).dt.year
df_train = y_train.join(train_year)
df_test = y_test.join(test_year)

vitals = ['HeartRate_Mean', 'SysBP_Mean', 'DiasBP_Mean', 'TempC_Max', 'RespRate_Mean', 'Glucose_Mean']
labs = ['ANIONGAP_max', 'ALBUMIN_max', 'BANDS_max',  'SODIUM_max', 'BUN_max', 'WBC_min']

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1(children='Exploratory Data Analysis'), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Visualising distribution and correlation of our data.'), className="mb-4")
        ]),
# choose between cases or deaths
    dcc.Dropdown(
        id='train_test',
        options=[
            {'label': 'Training Dataset', 'value': 'train'},
            {'label': 'Testing Dataset', 'value': 'test'},
        ],
        value='train',
        #multi=True,
        style={'width': '50%'}
        ),
# for some reason, font colour remains black if using the color option
    dbc.Row([
        dbc.Col(dbc.Card(html.H3(children='Target Variable: Discharge Locations',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='pie_target'), width=5),
        dbc.Col(dcc.Graph(id='line_target'), width=7)
        ]),

    dbc.Row([
        dbc.Col(dbc.Card(html.H3(children='Vitals and Laboratory Test Results',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
                , className="mt-4 mb-4")
    ]),

    dbc.Row([
        dbc.Col(html.H5(children='Correlation between Vitals and Labs', className="text-center"),
                width=5, className="mt-4"),
        dcc.Dropdown(
            id='vital_column',
            options=[
                {'label': 'Heart Rate', 'value': 'HeartRate_Mean'},
                {'label': 'Systolic Blood Pressure', 'value': 'SysBP_Mean'},
                {'label': 'Diastolic Blood Pressure', 'value': 'DiasBP_Mean'},
                {'label': 'Temprature (Celsius)', 'value': 'TempC_Max'},
                {'label': 'Respiratory Rate', 'value': 'RespRate_Mean'},
                {'label': 'Glucose', 'value': 'Glucose_Mean'},
                {'label': 'Anion Gap', 'value': 'ANIONGAP_max'},
                {'label': 'Albumin', 'value': 'ALBUMIN_max'},
                {'label': 'BANDS', 'value': 'BANDS_max'},
                {'label': 'Sodium', 'value': 'SODIUM_max'},
                {'label': 'BUN', 'value': 'BUN_max'},
                {'label': 'White Blood Count', 'value': 'WBC_min'},
            ],
            value='HeartRate_Mean',
            # multi=True,
            style={'width': '50%'}
        ),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='cor_vital'), width=5),
        dbc.Col(dcc.Graph(id='hist_vital'), width=7)
    ]),
])


])
target_label = ['HOME','SNF','Other Facility','Dead/Expired']
column_names = ['Hear Rate','Sys BP', 'Dias BP', 'Temperature', 'Respiratory Rate', 'Glucose', 'Anion Gap', 'Albumin', 'Bands', 'Sodium', 'BUN', 'White Blood Count']

@app.callback([Output('pie_target', 'figure'),
               Output('line_target', 'figure'),
               Output('cor_vital', 'figure'),
               Output('hist_vital', 'figure')],
              [Input('train_test', 'value'),
               Input('vital_column', 'value')])
def update_graph(train_test, vital_column):

    if train_test=="train":
        df_fig = y_train
        df_fig2 = df_train.groupby(['target', 'DISCHTIME'], as_index=False).size()
        df_fig3 = X_train.loc[:,vitals+labs].corr()
        df_fig4 = X_train.join(y_train)
    else:
        df_fig = y_test
        df_fig2 = df_test.groupby(['target', 'DISCHTIME'], as_index=False).size()
        df_fig3 = X_test.loc[:,vitals+labs].corr()
        df_fig4 = X_test.join(y_test)

    fig = go.Figure(data=[
        go.Pie(labels=target_label, values= df_fig.value_counts().values)
        ])

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      template = "seaborn",
                      margin=dict(t=0))
    fig2 = go.Figure()
    for i in range(1,5):
        df_fig2_filtered = df_fig2[df_fig2.target==i]
        fig2.add_trace(go.Scatter(x=df_fig2_filtered.DISCHTIME, y=df_fig2_filtered.iloc[:,2],
                                 name=target_label[i-1],
                                 mode='markers+lines'))

    fig2.update_layout(yaxis_title='Number of Observations',
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       margin=dict(t=0))

    fig3 = go.Figure(data=go.Heatmap(z=df_fig3, x=column_names, y=column_names, colorscale='Viridis'))

    fig4 = go.Figure()
    for i in range(1, 5, 1):
        df4_filtered = df_fig4[df_fig4.target == i]
        df4_filtered = df4_filtered[df4_filtered.loc[:,vital_column] != df4_filtered.loc[:,vital_column].mode().values[0]]
        fig4.add_trace(go.Histogram(x=df4_filtered.loc[:,vital_column], opacity=0.4, name=target_label[i - 1]))

    return fig, fig2, fig3, fig4

# needed only if running this as a single page app
#if __name__ == '__main__':
#    app.run_server(host='127.0.0.1', debug=True)