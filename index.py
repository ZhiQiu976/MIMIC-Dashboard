import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash

from app import app, server
# import all pages in the app
from apps import home, EDA, calendar, Models, deploy_app

#external_stylesheets = [dbc.themes.LUX]
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home",
                             href="/haha"
                             ),
        dbc.DropdownMenuItem("EDA",
                             href="/EDA"
                             ),
        dbc.DropdownMenuItem("Models",
                             href="/Models"
                             ),
        dbc.DropdownMenuItem("Calendar",
                                    href="/calendar"
                                    ),
        dbc.DropdownMenuItem("Online ML",
                                    href="/deploy_app"
                                    )
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        #dbc.Col(html.Img(src="/assets/virus.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("MIMIC Predictive Modelling", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

#def toggle_navbar_collapse(n, is_open):
#    if n:
#        return not is_open
#    return is_open

#for i in [2]:
#    app.callback(
#        Output(f"navbar-collapse{i}", "is_open"),
#        [Input(f"navbar-toggler{i}", "n_clicks")],
#        [State(f"navbar-collapse{i}", "is_open")],
#    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/EDA':
        return EDA.layout
    elif pathname == '/Models':
        return Models.layout
    elif pathname == '/calendar':
        return calendar.layout
    elif pathname == '/deploy_app':
        return deploy_app.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)