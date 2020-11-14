import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash.dependencies import Input, Output

import calendar
import datetime
from datetime import datetime
import pandas as pd
import numpy as np
import copy

from app import app

y_test = pd.read_csv("./Data/y_test.csv",index_col=0)
test_time = pd.read_csv("./Data/test_time.csv", index_col=0)
y_pred = pd.read_csv('./Data/test_results_decoded.csv', index_col=0).loc[:,'test_pred_xgb']

df = test_time.join(y_pred)
df['DISCHTIME'] = pd.to_datetime(df.DISCHTIME)
df['Year'] = df.DISCHTIME.dt.year
df['Month'] = df.DISCHTIME.dt.month
df['day'] = df.DISCHTIME.dt.day
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
            {'label': '2105', 'value': '2105'}
        ],
        value='2101',
        #multi=True,
        style={'width': '80%'}
        )),
        dbc.Col(dcc.Dropdown(
            id='month',
            options=[
                {'label': 'January', 'value': '1'},
                {'label': 'February', 'value': '2'},
                {'label': 'March', 'value': '3'},
                {'label': 'April', 'value': '4'},
                {'label': 'May', 'value': '5'},
                {'label': 'June', 'value': '6'},
                {'label': 'July', 'value': '7'},
                {'label': 'August', 'value': '8'},
                {'label': 'September', 'value': '9'},
                {'label': 'October', 'value': '10'},
                {'label': 'November', 'value': '11'},
                {'label': 'December', 'value': '12'},
            ],
            value='1',
            # multi=True,
            style={'width': '80%'}
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

months = ['January','February','March','April','May','June','July','August','September','October','Novemver','December']

@app.callback(Output('calendar', 'figure'),
              [Input('year', 'value'),
               Input('dischloc', 'value'),
               Input('month', 'value')])
def update_graph(year, dischloc, month):
    calendar_object = calendar.Calendar()
    df_year = df[df['Year'] == int(year)]
    df_month = df_year[df_year['Month'] == int(month)]
    df_selected = df_month[df_month['target'] == dischloc]

    date_string_array = df_selected.groupby('DISCHTIME').size().index.tolist()
    pp_array = df_selected.groupby('DISCHTIME').size().tolist()

    days1 = calendar_object.monthdatescalendar(int(year), int(month))
    days = []
    for i in range(len(days1)):
        days.append([None] * 7)
    for rows_number, rows in enumerate(days1):
        for time_index, time in enumerate(rows):
            days[rows_number][time_index] = time.day

    x = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    y = []
    for i in range(len(days1)):
        y.append('Week'+str(i+1))
    y = y[::-1]  ##REVERSE THE lists order
    day_numbers = len(date_string_array)
    dates_list = date_string_array

    color_array = copy.deepcopy(days1)
    textinfo = copy.deepcopy(days1)

    if day_numbers==0:
        for rows_number, rows in enumerate(days1):
            for time_index, time in enumerate(rows):
                color_array[rows_number][time_index] = 0
                textinfo[rows_number][time_index] = 'No patients being discharged to ' + 'HOME'
    else:
        for rows_number, rows in enumerate(days1):
            for time_index, time in enumerate(rows):
                for i in range(day_numbers):
                    if dates_list[i] == time:
                        if pp_array[i] >= 1:
                            color_array[rows_number][time_index] = pp_array[i]
                            textinfo[rows_number][time_index] = 'Number of patients discharged: ' + str(pp_array[i])
                            break
                    else:
                        color_array[rows_number][time_index] = 0
                        textinfo[rows_number][time_index] = 'No patients being discharged to ' + 'HOME'

    ## Z values indicate the color of date'cells.
    z = color_array[::-1]

    textinfo = textinfo[::-1]
    if day_numbers==0:
        colorscale = [[0.0, 'rgb(255,255,255)'],
                      [1.0, 'rgb(255, 255, 255)']]
    else:
        colorscale = [[0.0, 'rgb(255,255,255)'], [.25, 'rgb(255, 255, 153)'],
                      [.5, 'rgb(153, 255, 204)'], [0.75, 'rgb(179, 217, 255)'],
                      [1.0, 'rgb(240, 179, 255)']]

    fig = go.Figure(data= ff.create_annotated_heatmap(z, x=x, y=y, text=textinfo, hoverinfo='text', colorscale=colorscale))
    fig.update_layout(
        title={
            'text': months[int(month)-1],
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    return fig
