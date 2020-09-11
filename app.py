# 텐서플로 로드
import tensorflow.keras as keras
import tensorflow as tf
# Scailer load 용 라이브러리
import joblib
from sklearn.preprocessing import minmax_scale

# Web Page 관련 라이브러
from flask import Flask, request, redirect, url_for
from flask import render_template,Markup
from werkzeug.utils import secure_filename

# Map 관련 라이브러리
import folium
from folium.plugins import MarkerCluster
from flask_table import Table, Col

import pandas as pd
import numpy as np

import shutil
import os
import requests, json, sys
import datetime as dt
import time


app = Flask(__name__, template_folder='template')
# app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True

m = folium.Map(
    location=[36.5053542, 127.7043419],
    zoom_start=8
)

marker_cluster = MarkerCluster().add_to(m)
g_testCombine = ""
grad_index_columns = 0
grad_index_rows = 0
#전역변수 지정
# g_call_id = ""
# g_testCombine = pd.DataFrame("")
# g_testresult = pd.DataFrame("")

app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# pandas map background color 지정 함수
import math
import decimal
def background_color(val, grad_cam, total_df):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    global grad_index_rows
    global grad_index_columns
    grad_index_rows = grad_index_rows + 1
    if grad_index_rows-1 >= (grad_cam.shape[0] * 3):
        temp_index = (grad_index_rows-1) - (grad_cam.shape[0] * 3)
        temp_i = (grad_index_rows-1) % grad_cam.shape[0]
        temp_b = grad_index_columns
        temp_i = round(temp_i, 0)
        temp_i=math.trunc(temp_i)
        temp_b=math.trunc(temp_b)
        if temp_i == grad_cam.shape[0]-1:
            grad_index_columns = grad_index_columns + 1
        # print(temp_i, temp_b, grad_index_rows , val, total_df.iloc[temp_i]['data_ftp_rx_tp'])
        if temp_b > grad_cam.shape[1]-1:
            color = 'black'
            return 'background-color: %s' % color
                   
        if total_df.iloc[temp_i]['data_ftp_rx_tp'] > 200:
            color = 'black'
            return 'background-color: %s' % color
        
        grad_val=grad_cam[temp_i][temp_b]
        grad_cam_index = grad_cam[temp_i]
        c = grad_val
        diff_c = grad_cam_index.mean()
        
        if temp_b == 1:
            if val > 5:
                color = 'black'
                return 'background-color: %s' % color
        elif temp_b == 0:
            if val > -95:
                color = 'black'
                return 'background-color: %s' % color
        
        elif temp_b == 4:
            if val < 15:
                color = 'black'
                return 'background-color: %s' % color
        elif temp_b == 3:
            if val < 15:
                color = 'black'
                return 'background-color: %s' % color
        elif temp_b == 5:
            if val > 12:
                color = 'black'
                return 'background-color: %s' % color
        elif temp_b == 6:
            if val > 185:
                color = 'black'
                return 'background-color: %s' % color
        elif temp_b == 7:
            if val > 15:
                color = 'black'
                return 'background-color: %s' % color
        if diff_c <= c:
            color = 'background-color: red' if grad_val else 'black'
            return color
        elif diff_c/2 <= c:
            color = 'background-color: Orange' if grad_val else 'black'
            return color
            
    color = 'black'
    return 'background-color: %s' % color
# pandas map background color

# 업로드 된 파일을 받아서 upload_file csv 생성 및 main makrer 생성
def make_html(filename):
     
    test_file = r'C://Work//AI_Data_road_test//202007_NQI_도로_수도권_KT_NR5G_FTP_DL.csv'
    route =  'C:/Work/flask/cssfile/' + secure_filename(filename)
    testCombine = pd.read_csv(route, encoding='cp949')
    testCombine.loc[testCombine['total_call_state'] == 'T-3초 계산구간', 'total_call_state' ]= 'Traffic-3'
    testCombine.to_csv('C:/Work/flask/cssfile/upload_file.csv')
    test_result = testCombine[testCombine['total_call_state'] == 'Traffic-3']
    test_result = test_result.drop_duplicates(subset='call_id', keep = 'first')
    test_result = test_result[['latitude', 'longitude', 'call_id', 'data_ex_rx_thravg']]
    test_result.reset_index(drop=True, inplace=True)
        
    m = folium.Map(
        location=[test_result.iloc[0]['latitude'], test_result.iloc[0]['longitude']],
        zoom_start=12,
        tiles='Stamen Terrain'
    )   
    count = 0; 
    for i in range(len(test_result)):
        if(test_result.iloc[i]['data_ex_rx_thravg'] < 200):
            # Fail Call Marker 최대 300개
            if(count  > 300):
                break;
            count=count+1
            folium.Marker(
                [test_result.iloc[i]['latitude'], test_result.iloc[i]['longitude']],
                popup=test_result.iloc[i]['call_id'],
                tooltip="Call_THP = "  + str(test_result.iloc[i]['data_ex_rx_thravg']),
                icon=folium.Icon(color='red',icon='ok')).add_to(m)
    m.save('C:\\Work\\flask\\template\\marker.html')
####################
    
# 예측 된 결과의 MAP 표현
def make_predict_map_html(testCombine):
    m = folium.Map(
        location=[testCombine.iloc[0]['latitude'], testCombine.iloc[0]['longitude']],
        zoom_start=16,
    )
    testCombine.reset_index(drop=True, inplace=True)
    count = 0; 
    for i in range(len(testCombine)):
        if(testCombine.iloc[i]['data_ftp_rx_tp'] < 200 ):
            location = [testCombine.iloc[i]['latitude'], testCombine.iloc[i]['longitude']]
            folium.CircleMarker(location=location, radius=15, color='red', fill=True, fill_color='white').add_to(m)
            folium.Marker(location=location,
                        tooltip=testCombine.iloc[i]['top_match'],
                        popup=testCombine.iloc[i]['top_match'],
                        icon=folium.DivIcon(icon_size=(100,36), 
                        icon_anchor=(5,5),
                        html='<div style="font-size: 8pt; color : black">{}</div>'.format(testCombine.iloc[i]['top_match']))).add_to(m)
            m.add_child(folium.CircleMarker(location, radius=10))
    m.save('C:\\Work\\flask\\template\\predict_marker.html')

# 예측을 위한 가중치 환산
def generate_gradcam(img_tensor, model, class_index, activation_layer):
    y_c = model.outputs[0].op.inputs[0][0, class_index]
    A_k = model.get_layer(activation_layer).output
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(activation_layer).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_tensor])
        loss = predictions[:, 1]

    conv_output = conv_outputs[0]
    grad_val = tape.gradient(loss, conv_outputs)[0]
    weights = np.mean(grad_val, axis=(0, 1))
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]
    grad_cam = np.maximum(grad_cam, 0)
    return grad_cam, weights

# 가중치 base 의 Top 1,2,3 선정 및 AI Top Reason 산출
def top_match(merge_sample):
    grad_cam_data = merge_sample[['pre_rsrp', 'pre_sinr', 'pre_Txpwr', 'pre_dl_bler', 'pre_ul_bler', 'pre_dl_mcs', 'pre_dl_rb','pre_ul_mcs']]
    temp_result = []
    for index ,temp in enumerate(np.array(grad_cam_data)):
        c = temp.argsort()[-3:][::-1]
        c = c.reshape(1, 3)
        temp_result.append(c)
    c = np.array(temp_result)
    a = c.reshape(-1,3)
    a = pd.DataFrame(a, columns=['top1', 'top2', 'top3'])
    result_list = []
    for index, temp in enumerate(np.array(a)):
        for j_index in range(3):
            if(temp[j_index] == 0):
                if(merge_sample.iloc[index]['nr5g_Rsrp'] < -95):
                    result_list.append(temp[j_index])
                    break;
            elif(temp[j_index] == 1):
                if(merge_sample.iloc[index]['nr5g_Sinr'] < 5):
                    result_list.append(temp[j_index])
                    break;
            elif(temp[j_index] == 2):
                if(merge_sample.iloc[index]['nr5g_TxPwrTotal_Actual'] > 10):
                    result_list.append(temp[j_index])
                    break;
            elif(temp[j_index] == 3):
                if(merge_sample.iloc[index]['nr5g_Pdsch_Bler'] > 10):
                    result_list.append(temp[j_index])
                    break;
            elif(temp[j_index] == 4):
                if(merge_sample.iloc[index]['nr5g_pc_pusch_bler'] > 10):
                    result_list.append(temp[j_index])
                    break;
            elif(temp[j_index] == 5):
                if(merge_sample.iloc[index]['nr5g_DlMcs_Ant0_Avg'] < 10):
                    result_list.append(temp[j_index])
                    break;
            elif(temp[j_index] == 6):
                if(merge_sample.iloc[index]['nr5g_Rx_NumOfRb_Avg'] < 185):
                    result_list.append(temp[j_index])
                    break;
            elif(temp[j_index] == 7):
                if(merge_sample.iloc[index]['nr5g_UlMcs_Avg'] < 10):
                    result_list.append(temp[j_index])
                    break;
            if(j_index == 2):
                result_list.append(8)
    result_list = pd.DataFrame(result_list, columns=["top_match"])
    return pd.concat([result_list, a], axis=1)

# 모델을 통한 콜 ID 별 예측 및 Return Call
def make_predict_df(path):
    model = tf.keras.models.load_model(r'C://Work//flask//static//model//0901_200Mbps_val_96%.h5')
    scaler = joblib.load(r'C://Work//flask//static//model//scaler.save')
    testCombine = pd.read_csv('C:/Work/flask/cssfile/upload_file.csv', encoding='cp949')
    test_result = testCombine[testCombine['total_call_state'] == 'Traffic-3']
    test_result.reset_index(drop=True, inplace=True)
    test_list = test_result[['call_id','latitude', 'longitude','serving_network','sampled_time','call_type','total_call_state','nr5g_PCI','nr5g_Rsrp','nr5g_Sinr','nr5g_TxPwrTotal_Actual','nr5g_Pdsch_Bler','nr5g_pc_pusch_bler','nr5g_DlMcs_Ant0_Avg','nr5g_Rx_NumOfRb_Avg','nr5g_UlMcs_Avg','nr5g_Rx_NumOfRB','data_ex_rx_thravg','data_ftp_rx_tp']]
    test_list.loc[ : ,'result'] = 0
    test_list.loc[test_list['data_ftp_rx_tp'] < 200, 'result'] = 1
    table_df = test_list[test_result['call_id'] == path]
    test_drop_list = table_df.drop(['call_id', 'latitude', 'longitude','sampled_time' ,'serving_network','call_type','total_call_state', 'nr5g_PCI', 'nr5g_Rx_NumOfRB','result','data_ftp_rx_tp','data_ex_rx_thravg'],axis=1)
    test_drop_scale = scaler.transform(test_drop_list)
    test_drop_reshape = test_drop_scale.reshape(-1,1,8,1)
    conv_name = 'conv2d_5'
    grad_cam_total = []
    for i in range(len(test_drop_scale)):
        grad_cam_data = test_drop_scale[i]
        grad_cam_data = grad_cam_data.reshape(-1,1,8,1)
        test_pred = model.predict(grad_cam_data)
        grad_cam, grad_val = generate_gradcam(grad_cam_data, model , 1, conv_name)
        grad_cam_total.append(grad_cam)
    grad_cam_total =np.array(grad_cam_total).reshape(-1,8)
    temp=pd.DataFrame(grad_cam_total, columns=['pre_rsrp', 'pre_sinr', 'pre_Txpwr', 'pre_dl_bler', 'pre_ul_bler', 'pre_dl_mcs', 'pre_dl_rb','pre_ul_mcs'])
    table_df.reset_index(drop=True, inplace=True)
    merge_sample = pd.concat([table_df, temp],axis=1)
    return merge_sample, temp

# 그래프 코드
def getSeries(df,option):
    series = '['
    series += '{"name":"' + option + '","data":['
    for index, row in df.iterrows():
        if(np.isnan(row[option])):
            series += '[' + str(time.mktime(dt.datetime.strptime(row['sampled_time'], '%Y-%m-%d %H:%M:%S').timetuple())) + ',' + 'null' + '],'
        else:
            series += '[' + str(time.mktime(dt.datetime.strptime(row['sampled_time'], '%Y-%m-%d %H:%M:%S').timetuple())) + ',' + str(int(row[option])) + '],'
    series = series[:-1]
    series += ']},'
    series = series[:-1] + ']'
    return series

@app.route('/graph')
# @app.route('/analysis/graph/<path>')
@app.route('/analysis/graph/<path>')
def graph(path):
    testCombine = pd.read_csv('C:/Work/flask/cssfile/upload_file.csv', encoding='cp949')
#     test_result = testCombine[testCombine['total_call_state'] == 'T-3초 계산구간']
    test_result = testCombine[testCombine['total_call_state'] == 'Traffic-3']
    test_result.reset_index(drop=True, inplace=True)
    table_df = test_result[test_result['call_id'] == path]
    renderTo = ['nr5g_Rsrp','nr5g_Sinr','nr5g_TxPwrTotal_Actual','nr5g_Pdsch_Bler','nr5g_pc_pusch_bler','nr5g_DlMcs_Ant0_Avg','nr5g_Rx_NumOfRb_Avg','nr5g_UlMcs_Avg', 'nr5g_Rx_NumOfRB']
    option =['nr5g_Rsrp','nr5g_Sinr','nr5g_TxPwrTotal_Actual','nr5g_Pdsch_Bler','nr5g_pc_pusch_bler','nr5g_DlMcs_Ant0_Avg','nr5g_Rx_NumOfRb_Avg','nr5g_UlMcs_Avg', 'nr5g_Rx_NumOfRB']
    ttext=['nr5g_Rsrp','nr5g_Sinr','nr5g_TxPwrTotal_Actual','nr5g_Pdsch_Bler','nr5g_pc_pusch_bler','nr5g_DlMcs_Ant0_Avg','nr5g_Rx_NumOfRb_Avg','nr5g_UlMcs_Avg', 'nr5g_Rx_NumOfRB']
    ytext=['nr5g_Rsrp','nr5g_Sinr','nr5g_TxPwrTotal_Actual','nr5g_Pdsch_Bler','nr5g_pc_pusch_bler','nr5g_DlMcs_Ant0_Avg','nr5g_Rx_NumOfRb_Avg','nr5g_UlMcs_Avg', 'nr5g_Rx_NumOfRB']
    chartInfo = []
    chart_type = 'line'
    chart_height = 200
    for i in range(9):
        chart = {"renderTo": renderTo[i], "type": chart_type, "height": chart_height}
        series = getSeries(table_df, option[i])
        title = {"text":ttext[i]}
        xAxis = {"type":"datetime"}
        yAxis = {"title":{"text":ytext[i]}}
        chartInfo.append([chart, series, title, xAxis, yAxis])
#         chartInfo2.app
    
    return render_template('graph.html', chartInfo=chartInfo,)
# 그래프코드 종료
@app.route('/')
def index_html():
    return render_template('index.html', filename="",
                               Map_Info = "맵에 대한 설명",
                               open_page = "http://localhost:5000/analysis/123.html")

@app.route('/<path>')
def click_location(path):
    testCombine = pd.read_csv('C:/Work/flask/cssfile/upload_file.csv', encoding='cp949')
    test_result = testCombine[testCombine['total_call_state'] == 'Traffic-3']
    test_result.reset_index(drop=True, inplace=True)
    table_df = test_result[test_result['call_id'] == path]
    return render_template('index.html', filename="",
                               image_file = "image/1.jpg",
                               Map_Info = "맵에 대한 설명",
                               main_table = Markup(table_df.to_html(classes='table table-striped', 
                                                index=False, justify='center', header=True)),
                               open_page = "http://localhost:5000/analysis/"+path)

@app.route('/map/test.html')
def map_html():
    return render_template('marker.html')

@app.route('/analysis/map/<path>')
def analysis_map_html(path):
    return render_template('predict_marker.html')

@app.route('/fileUpload', methods=['GET', 'POST'])
def FileUload():
    if request.method == 'POST':
        f = request.files['file']
        #저장할 경로 + 파일명
        route =  'C:/Work/flask/cssfile/'
        f.save(route + secure_filename(f.filename))
        make_html(f.filename)
    
    return render_template('index.html', filename="아래 사진 영역에 MAP 보여 주기",
                           image_file = "image/1.jpg",
                           Map_Info = "맵에 대한 설명",
                           main_table = "위 지도에서 콜을 선택하세요.",
                          )
 
## 분석 페이지의 Main Page
@app.route('/analysis/<path>')
def analysis_html(path):
    df_raw, grad_cam = make_predict_df(path)
    df_predict_ranking = top_match(df_raw)
    df = pd.concat([df_raw, df_predict_ranking], axis=1)
    df = make_predict_map_html(df)
    df_raw.reset_index(drop=True, inplace=True)
    table_df = df_raw[df_raw['call_id'] == path]
    test_list = table_df[['serving_network','nr5g_PCI','nr5g_Rsrp','nr5g_Sinr','nr5g_TxPwrTotal_Actual','nr5g_Pdsch_Bler','nr5g_pc_pusch_bler','nr5g_DlMcs_Ant0_Avg','nr5g_Rx_NumOfRb_Avg','nr5g_UlMcs_Avg','nr5g_Rx_NumOfRB', 'data_ex_rx_thravg','data_ftp_rx_tp']]
    
    global grad_index_rows
    global grad_index_columns
    grad_index_columns = 0
    grad_index_rows = 0
    
    s = test_list.style.applymap(background_color, 
                                 grad_cam = np.array(grad_cam),
                                 total_df = table_df).highlight_null('black')
    
    return render_template('analysis-temp.html',
                           main_result ='SINR불량으로 인한 ~~~~~~~~',
                           analysis_table = Markup(s.render()),
                           graph_path="/analysis/graph/"+path,
                           map_path="/analysis/map/"+path)

if __name__ == '__main__':
    # threaded=True 로 넘기면 multiple plot이 가능해짐
    app.run(debug=True, threaded=True)