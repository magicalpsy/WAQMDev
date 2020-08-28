from flask import Flask, request, redirect, url_for
from flask import render_template,Markup
from werkzeug.utils import secure_filename
import shutil
import os
app = Flask(__name__, template_folder='template')
# app = Flask(__name__)

@app.route('/')
def index_html():
    call_id = "그래프 표시"
    return render_template('index.html', filename="", afteruplaod= "아래에 완성된 페이지 사용법", map_po = Markup('<div class="fakeimg">Fake Image</div>'))

@app.route('/fileUpload', methods=['GET', 'POST'])
def FileUload():
    if request.method == 'POST':
        f = request.files['file']
        #저장할 경로 + 파일명
        route =  'C:/Work/flask/cssfile/'
        f.save(route + secure_filename(f.filename))
    return render_template('index.html', filename="아래 사진 영역에 MAP 보여 주기",
                           afteruplaod="아래 부분에 지도 출력",
                           map_po = Markup('<div class="fakeimg"> Fake Image</div>'),
                           image_file = "image/1.jpg",
                           Map_Info = "맵에 대한 설명")

