import dash_html_components as html
import dash_bootstrap_components as dbc
import dash
# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

from app import app

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the MIMIC Predictive Modelling dashboard", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='This dashboard shows you the EDA and Modelling process of predicting dicharge types of patients '
                                     )
                    , className="mb-4")
            ]),

        dbc.Row([
            dbc.Col(html.H5(children='The type of discharge location of patients is a important representation of healthcare utilizations. '
                                     'Here, we use the demographics, vitals and laboratory data to predict four different types of discharge locations'
                                     )
                    , className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Link for original dataset',
                                               className="text-center"),
                                       dbc.Button("MIMIC",
                                                  href="https://mimic.physionet.org/",
                                                  color="primary",className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='Source code of this dashboard',
                                               className="text-center"),
                                       dbc.Button("GitHub-Repo 1",
                                                  href="https://github.com/biostats823-final-project/MIMIC-Dashboard",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='Ssource code of this project',
                                               className="text-center"),
                                       dbc.Button("Github-Repo 2",
                                                  href="https://github.com/biostats823-final-project/MIMIC-Predictive-Modeling",
                                                  color="primary",
                                                  className="mt-3"),

                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4")
        ], className="mb-5")

    ])

])

# needed only if running this as a single page app
#if __name__ == '__main__':
#    app.run_server(debug=True)
