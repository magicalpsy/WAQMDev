import pandas as pd
import numpy as np
from flask import Flask, request, redirect, url_for
from flask import render_template,Markup
from werkzeug.utils import secure_filename
from tqdm import tqdm
from folium.plugins import MarkerCluster
from flask_table import Table, Col
import folium
import shutil
import os
import requests, json, sys

app = Flask(__name__, template_folder='template')
# app = Flask(__name__)
m = folium.Map(
    location=[36.5053542, 127.7043419],
    zoom_start=8
)

marker_cluster = MarkerCluster().add_to(m)

#전역변수 지정
g_call_id = ""
g_testCombine = ""

app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

base_url = "https://8oi9s0nnth.apigw.ntruss.com/corona19-masks/v1"

# 전연변수 활용: 자바스크립트 만으로 사용 설정 변경
def click_Marker(call_id):
#     global g_call_id
#     g_call_id = call_id
    return call_id


def make_html():
    m = folium.Map(
        location=[37.5356233, 126.943145],
        zoom_start=12,
        tiles='Stamen Terrain'
    )
    test_file = r'C://Work//AI_Data_road_test//202007_NQI_도로_수도권_KT_NR5G_FTP_DL.csv'
    testCombine = pd.read_csv(test_file, encoding='ISO-8859-1')
    global g_testCombine
    g_testCombine = testCombine
    test_result = testCombine[testCombine['total_call_state'] == 'T-3ÃÊ °è»ê±¸°£']
    test_result = test_result.drop_duplicates(subset='call_id', keep = 'first')
    test_result = test_result[['latitude', 'longitude', 'call_id', 'data_ex_rx_thravg']]
    test_result.reset_index(drop=True, inplace=True)
    for i in range(len(test_result)):
        if(i > 300):
            break;
        if(test_result.iloc[i]['data_ex_rx_thravg'] < 200):
            folium.Marker(
                [test_result.iloc[i]['latitude'], test_result.iloc[i]['longitude']],
                popup=click_Marker(test_result.iloc[i]['call_id']),
                tooltip=test_result.iloc[i]['call_id'],
                icon=folium.Icon(color='red',icon='ok'),
            ).add_to(m)
    m.save('C:\\Work\\flask\\template\\marker.html')
    
@app.route('/')
def index_html():
#     global g_call_id
#     m = folium.Map(
#         location=[37.5356233, 126.943145],
#         zoom_start=12,
#         tiles='Stamen Terrain'
#         )
#     tooltip = 'Click me!'
#     folium.Marker([37.4214563, 126.943145], popup= click_Marker('123456'), tooltip=tooltip).add_to(m)
#     folium.Marker([37.5356233, 126.943145], popup=click_Marker('333333'), tooltip=tooltip).add_to(m)
#     folium.Marker([37.5386917, 126.94104], popup=click_Marker('666666'), tooltip=tooltip).add_to(m)
#     m.save('C:\\Work\\flask\\template\\marker.html')
#     return render_template('marker.html')
    return render_template('index.html', filename="",
#                                image_file = "image/1.jpg",
                               Map_Info = "맵에 대한 설명",
                               open_page = "http://localhost:5000/analysis/123.html")

@app.route('/<path>')
def click_location(path):
    test_file = r'C://Work//AI_Data_road_test//202007_NQI_도로_수도권_KT_NR5G_FTP_DL.csv'
    testCombine = pd.read_csv(test_file, encoding='ISO-8859-1')
    test_result = testCombine[testCombine['total_call_state'] == 'T-3ÃÊ °è»ê±¸°£']
    test_result.reset_index(drop=True, inplace=True)
    table_df = test_result[test_result['call_id'] == path]
    return render_template('index.html', filename="",
                               image_file = "image/1.jpg",
                               Map_Info = "맵에 대한 설명",
                               main_table = Markup(table_df.to_html(classes='table table-striped', index=False, justify='center', header=True)), 
                               open_page = "http://localhost:5000/analysis/"+path)

@app.route('/map/test.html')
def map_html():
    return render_template('marker.html')

@app.route('/analysis/map/test.html')
def analysis_map_html():
    return render_template('marker.html')

@app.route('/fileUpload', methods=['GET', 'POST'])
def FileUload():
    if request.method == 'POST':
        f = request.files['file']
        #저장할 경로 + 파일명
        route =  'C:/Work/flask/cssfile/'
        f.save(route + secure_filename(f.filename))
    make_html()
    return render_template('index.html', filename="아래 사진 영역에 MAP 보여 주기",
                           image_file = "image/1.jpg",
                           Map_Info = "맵에 대한 설명",
#                            main_table = path, 
                          )
#                            open_page = "http://localhost:5000/analysis/123.html"+path)
#                            open_page = "http://localhost:5000/analysis/"+f.filename)

@app.route('/analysis/<path>')
def analysis_html(path):
    print(path)
    return render_template('analysis-temp.html', filename="")
#     return render_template('analysis-Test.html', filename="", image_file = "image/region.jpg")

def marker_color(col):
    col_dict = {
        'plenty' : 'green',
        'some' : 'orange',
        'few' : 'red',
        'empty' : 'gray'
    }
    return col_dict[col] if col in col_dict else 'black'


    
    
# from sklearn.externals import joblib
# scaler_filename = "scaler.save"
# joblib.dump(scaler, scaler_filename) 

# # And now to load...

# scaler = joblib.load(scaler_filename) 