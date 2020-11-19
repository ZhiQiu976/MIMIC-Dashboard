import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

from app import app

# needed if running single page dash app instead
# external_stylesheets = [dbc.themes.LUX]

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

y_train = pd.read_csv("./Data/y_train.csv", index_col=0)
y_test = pd.read_csv("./Data/y_test.csv", index_col=0)

X_train = pd.read_csv("./Data/X_train.csv", index_col=0)
X_test = pd.read_csv("./Data/X_test.csv", index_col=0)

train_time = pd.read_csv("./Data/train_time.csv", index_col=0)
test_time = pd.read_csv("./Data/test_time.csv", index_col=0)

train_year = pd.to_datetime(train_time.DISCHTIME).dt.year
test_year = pd.to_datetime(test_time.DISCHTIME).dt.year
df_train = y_train.join(train_year)
df_test = y_test.join(test_year)

X_train_dm = pd.read_csv("./Data/X_train_nodummy.csv", index_col=0)
X_test_dm = pd.read_csv("./Data/X_test_nodummy.csv", index_col=0)

dm = X_train_dm.iloc[:, 0:7]
vital = X_train_dm.iloc[:, 7:]
dm_y = pd.concat([dm.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
dm_y['target'] = dm_y['target'].replace({1: 'HOME', 2: 'SNF', 3: 'Other Facility', 4: 'Dead/Expired'})
vital_y = pd.concat([vital.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
vital_y['target'] = vital_y['target'].replace({1: 'HOME', 2: 'SNF', 3: 'Other Facility', 4: 'Dead/Expired'})

dm_test = X_test_dm.iloc[:, 0:7]
vital_test = X_test_dm.iloc[:, 7:]
dm_y_test = pd.concat([dm_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
dm_y_test['target'] = dm_y_test['target'].replace({1: 'HOME', 2: 'SNF', 3: 'Other Facility', 4: 'Dead/Expired'})
vital_y_test = pd.concat([vital_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
vital_y_test['target'] = vital_y_test['target'].replace({1: 'HOME', 2: 'SNF', 3: 'Other Facility', 4: 'Dead/Expired'})

vitals_label = ['HeartRate_Mean', 'SysBP_Mean', 'DiasBP_Mean', 'TempC_Max', 'RespRate_Mean', 'Glucose_Mean']
labs_label = ['ANIONGAP_max', 'ALBUMIN_max', 'BANDS_max', 'SODIUM_max', 'BUN_max', 'WBC_min']

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
            # multi=True,
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

        # add tables and visualizations of predictors
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Summary statistics of predictors',
                                     className="text-center text-light bg-dark"), body=True, color="dark")
                    , className="mt-4 mb-4")
        ]),

        # choose among different demographic predictors
        dcc.Dropdown(
            id='demographics',
            options=[
                {'label': 'Admission Type', 'value': 'ADMISSION_TYPE'},
                {'label': 'Admission Location', 'value': 'ADMISSION_LOCATION'},
                {'label': 'Insurance', 'value': 'INSURANCE'},
                {'label': 'Religion', 'value': 'RELIGION'},
                {'label': 'Gender', 'value': 'GENDER'},
                {'label': 'Ethnicity', 'value': 'ETHNICITY'},
                {'label': 'Diagnosis', 'value': 'DIAGNOSIS'},
            ],
            value='ADMISSION_TYPE',
            # multi=True,
            style={'width': '50%'},
            className="mt-4"
        ),

        dbc.Row([
            dbc.Col(html.H5(children='Categorical Variables', className="text-center"),
                    className="mt-4"),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='table_dm'))
        ]),

        dbc.Row([
            dbc.Col(html.H5(children='Stacked bar plot', className="text-center"),
                    className="mt-4"),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='bar_dm'), className="text-center")
        ]),

        dcc.Dropdown(
            id='vitals',
            options=[
                {'label': 'HeartRate Mean', 'value': 'HeartRate_Mean'},
                {'label': 'SysBP Mean', 'value': 'SysBP_Mean'},
                {'label': 'DiasBP Mean', 'value': 'DiasBP_Mean'},
                {'label': 'TempC Max', 'value': 'TempC_Max'},
                {'label': 'RespRate Mean', 'value': 'RespRate_Mean'},
                {'label': 'Glucose Mean', 'value': 'Glucose_Mean'},
                {'label': 'ICU:Length of stay', 'value': 'ICU_LOS'},
                {'label': 'Emergency department stay', 'value': 'EDstay'},
                {'label': 'Age', 'value': 'age'},
                {'label': 'Aniongap Min', 'value': 'ANIONGAP_min'},
                {'label': 'Aniongap Max', 'value': 'ANIONGAP_max'},
                {'label': 'Albumin Min', 'value': 'ALBUMIN_min'},
                {'label': 'Albumin Max', 'value': 'ALBUMIN_max'},
                {'label': 'Bands Min', 'value': 'BANDS_min'},
                {'label': 'Bands Max', 'value': 'BANDS_max'},
                {'label': 'Bicarbonate Min', 'value': 'BICARBONATE_min'},
                {'label': 'Bicarbonate Max', 'value': 'BICARBONATE_max'},
                {'label': 'Bilirubin Min', 'value': 'BILIRUBIN_min'},
                {'label': 'Bilirubin Max', 'value': 'BILIRUBIN_max'},
                {'label': 'Cheatitne Min', 'value': 'CREATININE_min'},
                {'label': 'Cheatitne Max', 'value': 'CREATININE_max'},
                {'label': 'Chloride Min', 'value': 'CHLORIDE_min'},
                {'label': 'Chloride Max', 'value': 'CHLORIDE_max'},
                {'label': 'Glucose Min', 'value': 'GLUCOSE_min'},
                {'label': 'Glucose Max', 'value': 'GLUCOSE_max'},
                {'label': 'Hematocrit Min', 'value': 'HEMATOCRIT_min'},
                {'label': 'Hematocrit Max', 'value': 'HEMATOCRIT_max'},
                {'label': 'Hemoglobin Min', 'value': 'HEMOGLOBIN_min'},
                {'label': 'Hemoglobin Max', 'value': 'HEMOGLOBIN_max'},
                {'label': 'Lactate Min', 'value': 'LACTATE_min'},
                {'label': 'Lactate Max', 'value': 'LACTATE_max'},
                {'label': 'Platelet Min', 'value': 'PLATELET_min'},
                {'label': 'Platelet Max', 'value': 'PLATELET_max'},
                {'label': 'Potassium Min', 'value': 'POTASSIUM_min'},
                {'label': 'Potassium Max', 'value': 'POTASSIUM_max'},
                {'label': 'Pulse Transit Time Min', 'value': 'PTT_min'},
                {'label': 'Pulse Transit Time Max', 'value': 'PTT_max'},
                {'label': 'International Normalised Ratio Min', 'value': 'INR_min'},
                {'label': 'International Normalised Ratio Max', 'value': 'INR_max'},
                {'label': 'Prothrombin Time Min', 'value': 'PT_min'},
                {'label': 'Prothrombin Time Max', 'value': 'PT_max'},
                {'label': 'Sodium Min', 'value': 'SODIUM_min'},
                {'label': 'Sodium Max', 'value': 'SODIUM_max'},
                {'label': 'Blood urea nitrogen Min', 'value': 'BUN_min'},
                {'label': 'Blood urea nitrogen Max', 'value': 'BUN_max'},
                {'label': 'White blood cells Min', 'value': 'WBC_min'},
                {'label': 'White blood cells Max', 'value': 'WBC_max'},
            ],
            value='HeartRate_Mean',
            # multi=True,
            style={'width': '50%'},
            className="mt-4"
        ),

        dbc.Row([
            dbc.Col(html.H5(children='Continuous Variables', className="text-center"),
                    className="mt-4"),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='table_vital'))
        ]),
        dcc.Graph(id='hist_vital'),

        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Correlation Plot',
                                     className="text-center text-light bg-dark"), body=True, color="dark")
                    , className="mt-4 mb-4")
        ]),

        dcc.Graph(id='cor_vital')
    ]),
])
target_label = ['HOME', 'SNF', 'Other Facility', 'Dead/Expired']
column_names = ['Hear Rate', 'Sys BP', 'Dias BP', 'Temperature', 'Respiratory Rate', 'Glucose', 'Anion Gap', 'Albumin',
                'Bands', 'Sodium', 'BUN', 'White Blood Count', 'Discharge Locations']


@app.callback([Output('pie_target', 'figure'),
               Output('line_target', 'figure'),
               Output('hist_vital', 'figure')],
              [Input('train_test', 'value'),
               Input('vitals', 'value')])
def update_graph(train_test, vitals):
    """Create pie plot of target vairbale, line plot of number of observations vs. discharge time,
       histogram of continuous variables

    Parameters
    ----------
    vitals : string
        Name of continuous predictor values, read from the dropdown
        
    train_test: string
        Train or test indicate using the training set or testing set
    
    Returns
    -------
    fig, fig2, fig4
    	Pie plot, line plot, histogram
    """
    if train_test is None:
        df_fig = y_train
        df_fig2 = df_train.groupby(['target', 'DISCHTIME'], as_index=False).size()
        df_fig4 = X_train.join(y_train)
    elif train_test == "train":
        df_fig = y_train
        df_fig2 = df_train.groupby(['target', 'DISCHTIME'], as_index=False).size()
        df_fig4 = X_train.join(y_train)
    else:
        df_fig = y_test
        df_fig2 = df_test.groupby(['target', 'DISCHTIME'], as_index=False).size()
        df_fig4 = X_test.join(y_test)

    fig = go.Figure(data=[
        go.Pie(labels=target_label, values=df_fig.value_counts().values)
    ])

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      template="seaborn",
                      margin=dict(t=0))
    fig2 = go.Figure()
    for i in range(1, 5):
        df_fig2_filtered = df_fig2[df_fig2.target == i]
        fig2.add_trace(go.Scatter(x=df_fig2_filtered.DISCHTIME, y=df_fig2_filtered.iloc[:, 2],
                                  name=target_label[i - 1],
                                  mode='markers+lines'))

    fig2.update_layout(yaxis_title='Number of Observations',
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       margin=dict(t=0))

    if vitals is None:
        vitals = 'HeartRate_Mean'
    fig4 = go.Figure()
    for i in range(1, 5, 1):
        df4_filtered = df_fig4[df_fig4.target == i]
        df4_filtered = df4_filtered[df4_filtered.loc[:, vitals] != df4_filtered.loc[:, vitals].mode().values[0]]
        fig4.add_trace(go.Histogram(x=df4_filtered.loc[:, vitals], opacity=0.4, name=target_label[i - 1]))

    return fig, fig2, fig4


@app.callback(Output('cor_vital', 'figure'),
              [Input('train_test', 'value')])
def update_correlation(train_test):
     """Create correlation plot

    Parameters
    ----------       
    train_test: string
        Train or test indicate using the training set or testing set
    
    Returns
    -------
    fig3
    	Corrrelation plot
    """
    if train_test == "train":
        df_fig3 = X_train.loc[:, vitals_label + labs_label].join(y_train).corr()
    else:
        df_fig3 = X_test.loc[:, vitals_label + labs_label].join(y_test).corr()
    fig3 = go.Figure(data=go.Heatmap(z=df_fig3, x=column_names, y=column_names, colorscale='Viridis'))
    return fig3


@app.callback([Output('table_dm', 'figure'),
               Output('bar_dm', 'figure')],
              [Input('demographics', 'value'),
               Input('train_test', 'value')])
def update_table_dm(demographics, train_test):
    """Create table and starked bar plot for all categorical predictors

    Parameters
    ----------
    demographics : string
        Name of predictor values, read from the dropdown
        
    train_test: string
        Train or test indicate using the training set or testing set
    
    Returns
    -------
    fig, fig2
    	Table and starked bar plot
    """
    if train_test == "test":
        dm_y_select = dm_y_test.copy()
    else:
        dm_y_select = dm_y.copy()
    if demographics == "ADMISSION_TYPE":
        df_dm = pd.crosstab(index=dm_y_select['ADMISSION_TYPE'], columns=dm_y_select['target'])
    elif demographics == "ADMISSION_LOCATION":
        df_dm = pd.crosstab(index=dm_y_select['ADMISSION_LOCATION'], columns=dm_y_select['target'])
    elif demographics == "INSURANCE":
        df_dm = pd.crosstab(index=dm_y_select['INSURANCE'], columns=dm_y_select['target'])
    elif demographics == "RELIGION":
        df_dm = pd.crosstab(index=dm_y_select['RELIGION'], columns=dm_y_select['target'])
    elif demographics == "GENDER":
        df_dm = pd.crosstab(index=dm_y_select['GENDER'], columns=dm_y_select['target'])
    elif demographics == "ETHNICITY":
        df_dm = pd.crosstab(index=dm_y_select['ETHNICITY'], columns=dm_y_select['target'])
    elif demographics == "DIAGNOSIS":
        df_dm = pd.crosstab(index=dm_y_select['DIAGNOSIS'], columns=dm_y_select['target'])
    elif demographics is None:
        df_dm = pd.crosstab(index=dm_y_select['ADMISSION_TYPE'], columns=dm_y_select['target'])

    fig = ff.create_table(df_dm, index=True, index_title=df_dm.index.name, annotation_offset=0.40)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      template="seaborn",
                      margin=dict(t=0))

    fig2 = px.histogram(dm_y_select, y=demographics, color="target", template="simple_white",
                        labels={"target": ""})

    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       template="seaborn",
                       margin=dict(t=0))

    return fig, fig2


@app.callback(Output('table_vital', 'figure'),
              [Input('vitals', 'value'),
               Input('train_test', 'value')])
def update_table_vital(vitals, train_test):
    """Create summary statistics table for all continuous predictors

    Parameters
    ----------
    vitals : string
        Name of predictor values, read from the dropdown
        
    train_test: string
        Train or test indicate using the training set or testing set
    
    Returns
    -------
    fig3
    	Table
    """
    global df_vital
    if train_test == "train":
        vital_y_select = vital_y.copy()
    else:
        vital_y_select = vital_y_test.copy()

    vital_columns = vital_y_select.columns
    temp = vital_y_select.groupby('target').describe()
    for i in range(0, 47):
        if vitals is None:
            vitals == vital_columns[1]
            break
        elif vitals == vital_columns[i]:
            df_vital = temp.loc[:, vitals].round(2)
            break
        else:
            continue
    fig3 = ff.create_table(df_vital, index=temp.index.all())
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       template="seaborn",
                       margin=dict(t=0))
    return fig3

# needed only if running this as a single page app
# if __name__ == '__main__':
#    app.run_server(host='127.0.0.1', debug=True)
