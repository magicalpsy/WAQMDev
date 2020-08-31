from flask import Flask, request, redirect, url_for
from flask import render_template,Markup
from werkzeug.utils import secure_filename
import shutil
import os
app = Flask(__name__, template_folder='template')
# app = Flask(__name__)

@app.route('/')
def index_html():
    return render_template('index.html', filename="")

@app.route('/fileUpload', methods=['GET', 'POST'])
def FileUload():
    if request.method == 'POST':
        f = request.files['file']
        #저장할 경로 + 파일명
        route =  'C:/Work/flask/cssfile/'
        f.save(route + secure_filename(f.filename))
    return render_template('index.html', filename="아래 사진 영역에 MAP 보여 주기",
                           image_file = "image/1.jpg",
                           Map_Info = "맵에 대한 설명",
                          open_page = "http://localhost:5000/analysis/"+f.filename)

@app.route('/analysis/<path>')
def analysis_html(path):
    print(path)
    return render_template('analysis-temp.html', filename="")
#     return render_template('analysis-Test.html', filename="", image_file = "image/region.jpg")