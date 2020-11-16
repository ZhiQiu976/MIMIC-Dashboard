try:
    import cPickle as pickle
except:
    import pickle
import joblib
import dash_bootstrap_components as dbc
import flask
import pandas as pd 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# external_stylesheets = [dbc.themes.LUX]

from app import app

def model_load(file): 
    with open(file, 'rb') as f:
        clf = joblib.load(f)   
    return clf

model = model_load("gbm_gamma.pickle")

val = [ 93.29166978, 119.24813733,  60.74150063,  37.50735481,
        18.67282058, 140.62989954,   5.03019082,  52.80626612,
        12.97661654,  15.5608457 ,   3.1575932 ,   3.24251062,
         6.08835911,   7.21947058,  22.48370555,  24.79604141,
         2.10361404,   2.36498379,   1.18063039,   1.4060783 ,
       102.88185862, 107.14042891, 112.52661905, 177.86146759,
        33.0118634 ,  38.35388185,  11.21510109,  12.85318605,
         1.83241933,   3.07279016, 213.94678703, 249.57208132,
         3.79854296,   4.69888232,  31.38408038,  43.31872765,
         1.3202494 ,   1.59110596,  14.47121372,  16.34227229,
       136.69684582, 140.15971052,  22.0233299 ,  26.08794345,
        11.31091528,  14.16679297,   0.        ,   0.        ,
         1.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         1.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   1.        ]

col = ['HeartRate_Mean', 'SysBP_Mean', 'DiasBP_Mean', 'TempC_Max',
       'RespRate_Mean', 'Glucose_Mean', 'ICU_LOS', 'age', 'ANIONGAP_min',
       'ANIONGAP_max', 'ALBUMIN_min', 'ALBUMIN_max', 'BANDS_min', 'BANDS_max',
       'BICARBONATE_min', 'BICARBONATE_max', 'BILIRUBIN_min', 'BILIRUBIN_max',
       'CREATININE_min', 'CREATININE_max', 'CHLORIDE_min', 'CHLORIDE_max',
       'GLUCOSE_min', 'GLUCOSE_max', 'HEMATOCRIT_min', 'HEMATOCRIT_max',
       'HEMOGLOBIN_min', 'HEMOGLOBIN_max', 'LACTATE_min', 'LACTATE_max',
       'PLATELET_min', 'PLATELET_max', 'POTASSIUM_min', 'POTASSIUM_max',
       'PTT_min', 'PTT_max', 'INR_min', 'INR_max', 'PT_min', 'PT_max',
       'SODIUM_min', 'SODIUM_max', 'BUN_min', 'BUN_max', 'WBC_min', 'WBC_max',
       'EDstay', 'ADMISSION_TYPE_ELECTIVE', 'ADMISSION_TYPE_EMERGENCY',
       'ADMISSION_TYPE_NEWBORN', 'ADMISSION_TYPE_URGENT',
       'ADMISSION_LOCATION_CLINIC REFERRAL/PREMATURE',
       'ADMISSION_LOCATION_EMERGENCY ROOM ADMIT',
       'ADMISSION_LOCATION_HMO REFERRAL/SICK',
       'ADMISSION_LOCATION_PHYS REFERRAL/NORMAL DELI',
       'ADMISSION_LOCATION_TRANSFER FROM HOSP/EXTRAM',
       'ADMISSION_LOCATION_TRANSFER FROM OTHER HEALT',
       'ADMISSION_LOCATION_TRANSFER FROM SKILLED NUR',
       'ADMISSION_LOCATION_TRSF WITHIN THIS FACILITY', 'INSURANCE_Government',
       'INSURANCE_Medicaid', 'INSURANCE_Medicare', 'INSURANCE_Private',
       'INSURANCE_Self Pay', '7TH DAY ADVENTIST', 'BUDDHIST', 'HINDU',
       'JEWISH', 'MUSLIM', 'ASIAN', 'BLACK', 'ETHNICITY_Others', 'WHITE',
       'ALTERED MENTAL STATUS', 'CHEST PAIN', 'CONGESTIVE HEART FAILURE',
       'CORONARY ARTERY DISEASE',
       'CORONARY ARTERY DISEASE\CORONARY ARTERY BYPASS GRAFT /SDA',
       'GASTROINTESTINAL BLEED', 'INTRACRANIAL HEMORRHAGE', 'PNEUMONIA',
       'SEPSIS', 'GENDER_F', 'GENDER_M']

tp = pd.DataFrame(val).transpose()
tp.columns=col


# server = flask.Flask(__name__)
# app = dash.Dash(__name__, server=server,external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Real Time Prediction Serverless App"), className="mb-2")

    ]) ]),
    # html.Div(children="Xu, Zhenhui - Master of Biostatistics",className="text-center"),
    # html.Div(children="Feng, Yuan - Master of Interdisciplinary Data Sciencse", className="text-center"),
    # html.Div(children="Qiu, Zhi (Heather) - Master of Statistical Science", className="text-center"),
    # html.Div(children="Han, Mengyue - Master of Biostatistics", className="text-center"),
    # html.Hr(),

    dbc.Container([
        dbc.Row([

                    dbc.Col(dbc.Card(html.H3(children='Feature Input',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
    ]) ]),

    dbc.Input(id="Age",
              type="number", 
              placeholder="Age"),
    
    dbc.Input(id="Gender",
              type="number", 
              placeholder="Binary Input: 1 = male, 0 = female"),
    
    
    dbc.Input(id="HeartRate_Mean",
              type="number", 
              placeholder="Mean heart rate"),
    
    #dcc
    dbc.Input(id="Glucose_Mean",
              type="number", 
              placeholder="Mean glucose level"),

    dbc.Input(id="TempC_Max",
              type="number", 
              placeholder="Mean temperature"),


    dbc.Input(id="INSURANCE_Medicare",
              type="number", 
              placeholder="inary Input: Insurance medicare = 1 else 0"),


    dbc.Input(id="ADMISSION_TYPE_EMERGENCY",
              type="number", 
              placeholder="Binary Input: emergency = 1 else = 0"),


    dbc.Input(id="ADMISSION_LOCATION",
              type="number", 
              placeholder="Binary Input: referral = 1 else = 0"),

    dbc.Input(id="INSURANCE_Private",
              type="number", 
              placeholder="Binary Input: private insurance = 1 else = 0"),

    dbc.Input(id="ADMISSION_TYPE_NEWBORN",
              type="number", 
              placeholder="Binary Input: newborn = 1 else = 0"),

    dbc.Row(dbc.Card(children=[html.H3(children='',
                                               className="text-center"),

                                       dbc.Button("Go",
                                                  id = "show",
                                                  color="primary",
                                                  className="mt-3"),

                                       ],
                             body=True, color="light", outline=False)
                    , className="mb-4"),

    #dbc.Button('Go', id='show',  color="primary", className="mt-3"),
    html.Div(id='out')]
)


@app.callback(
    Output("out", "children"),
    [Input("show", "n_clicks")],
    
    state=[State("Age", "value"),
           State("Gender", "value"),
           State("HeartRate_Mean", "value"),
           State("Glucose_Mean", "value"),
           State("TempC_Max", "value"),
           State("INSURANCE_Medicare", "value"),
           State("ADMISSION_TYPE_EMERGENCY", "value"),
           State("ADMISSION_LOCATION", "value"),
           State("INSURANCE_Private", "value"),
           State("ADMISSION_TYPE_NEWBORN", "value"),
           ])

def predict(n_clicks, sl, sw, pl, pw, var0,var1,var2,var3,var4,var5):
    if n_clicks is None:
        res = "No Input"
        return html.Div('Please input the values to see prediction.')
    else:
        flag = True
        for i in [sl, sw, pl, pw, var0,var1,var2,var3,var4,var5]:
            if i == None:
                flag = False
            if i in [sw,var1,var2,var3,var4,var5]:
                if i not in [0,1]:
                    flag = False
        if flag: 
            tp["HeartRate_Mean"] = pl
            tp["Glucose_Mean"] = pw
            tp["age"] = sl
            tp["GENDER_M"] = sw
            tp["TempC_Max"] = var0

            tp["INSURANCE_Medicare"] = var1
            tp["ADMISSION_TYPE_EMERGENCY"] = var2
            tp["ADMISSION_LOCATION_PHYS REFERRAL/NORMAL DELI"] = var3
            tp["INSURANCE_Private"] = var4
            tp["ADMISSION_TYPE_NEWBORN"] = var5
            pred = model.predict(tp)[0]
            res = 0

            if pred == 1:
                res = "Home"
                im = "https://hgtvhome.sndimg.com/content/dam/images/hgtv/fullset/2020/3/6/0/sh2020_front-yard-straight-lights-KB2A0223_edited_h.jpg.rend.hgtvcom.616.462.suffix/1583505586135.jpeg"
            elif pred == 2:
                res = "SNF"
                im = "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSTv8jeKO92saOWUK2W1y_dr1clTAJwL0ftLQ&usqp=CAU"
            elif pred == 3:
                res= "Other"
                im = "https://image.flaticon.com/icons/png/512/2115/2115958.png"
            else:
                res = "Dead/Expired"
                im =  "https://icon-library.com/images/dead-icon/dead-icon-17.jpg"
        else:
            res = "No sufficient/correct input, please re-fill the blanks"
            im = "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR361wqsab2SRwTjn6YFl-aOnlG9h8yGSUozg&usqp=CAU"

    return html.Div([
        html.Div('                Best guess according to your input: {}'.format(res)), 
        html.Hr(),
        html.Img(src=im ,style={'textAlign': 'center'}) #, style={'height':'20%', 'width':'20%'}
    ], style={'textAlign': 'center'})
    
    

# if __name__ == '__main__':
#     app.run_server(debug=True)