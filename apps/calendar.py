import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

from app import app

y_test = pd.read_csv("./Data/y_test.csv",index_col=0)
test_time = pd.read_csv("./Data/test_time.csv", index_col=0)
y_pred = pd.read_csv('./Data/test_results_decoded.csv', index_col=0).loc[:,'test_pred_xgb']

df = test_time.join(y_pred)
df['DISCHTIME'] = pd.to_datetime(df.DISCHTIME)
df['Year'] = df.DISCHTIME.dt.year
df = df.rename(columns={'test_pred_xgb':'target'})

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1(children='Calendar of Predicted Discharge Locations'), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Colored square represents predicted discharges to the selected location.'), className="mb-4")
        ]),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
        id='year',
        options=[
            {'label': '2101', 'value': '2101'},
            {'label': '2102', 'value': '2102'},
            {'label': '2103', 'value': '2103'},
            {'label': '2104', 'value': '2104'},
        ],
        value='2101',
        #multi=True,
        style={'width': '50%'}
        )),
        dbc.Col(dcc.Dropdown(
        id='dischloc',
        options=[
            {'label': 'HOME', 'value': 'HOME'},
            {'label': 'SNF', 'value': 'SNF'},
            {'label': 'Other Facility', 'value': 'OTHERS'},
            {'label': 'Dead/Expired', 'value': 'DEAD/EXPIRED'},
        ],
        value='SNF',
        #multi=True,
        style={'width': '80%'}
        ))
    ]),

    dcc.Graph(id='calendar'),
])
])

@app.callback(Output('calendar', 'figure'),
              [Input('year', 'value'),
               Input('dischloc', 'value')])
def update_graph(year, dischloc):
    df_year = df[df['Year'] == int(year)]
    df_selected = df_year[df_year['target'] == dischloc]
    by_day = (df_selected.groupby('DISCHTIME', as_index=False).size().
              rename(columns={'size': 'target'}).
              set_index('DISCHTIME').
              reindex(pd.date_range(start=year, end=str(int(year) + 1), freq='D')[:-1])
              )
    by_day = pd.DataFrame({'data': by_day.target,
                           'fill': 1,
                           'day': by_day.index.dayofweek,
                           'week': by_day.index.week})
    by_day.loc[(by_day.index.month == 1) & (by_day.week > 50), 'week'] = 0
    by_day.loc[(by_day.index.month == 12) & (by_day.week < 10), 'week'] = by_day.week.max() + 1

    plot_data = by_day.pivot('day', 'week', 'data').values[::-1]
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)

    colorscale = [[0.0, 'rgb(255,255,255)'], [1.0, 'rgb(240, 179, 255)'], [2.0, 'rgb(255, 77, 148)']]

    fig = go.Figure(data=go.Heatmap(
        z=plot_data,
        y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        #colorscale=colorscale,
        hoverongaps=False))
    fig.update_layout(
       title='Number of Patients Discharged to SNF',
       showlegend=False)
    fig.update_traces(showscale=False)

    return fig
