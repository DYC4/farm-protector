# app.py
import json

from flask import Flask, render_template, request, make_response

import dbModule

app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/')
def main_page():
  return render_template('index.html')


@app.route('/we-do')  ## 수정 해야함
def we_do():
  return render_template('intro1.html')


@app.route('/we-are')  ## 수정 해야함
def we_are():
  return render_template('intro2.html')


@app.route('/our-service')  ## 수정 해야함
def our_service():
  return render_template('upload.html')

@app.route('/test', methods=['GET', 'POST'])  ## 수정 해야함
def test():
  print(request.files['myfile'])
  # 사진처리하는방법: https://roksf0130.tistory.com/124
  category = request.form["plant"]
  bug = "점박이응애"#사진 분석 한걸 넣기
  db_class = dbModule.Database()
  sql = 'SELECT * FROM list WHERE plant = "{}" AND name = "{}" limit 10;'.format(category, bug)
  print(sql)
  row = db_class.executeALL(sql)
  print(row)
  result = dict()
  for i in range(len(row)):
    result[i] = row[i]
  resp = make_response(json.dumps(result, ensure_ascii=False))

  return render_template('result.html', bug=bug, data=row)


if __name__=="__main__":
  app.run(debug=True)