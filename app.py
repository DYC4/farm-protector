# app.py
from flask import Flask, render_template, request

app = Flask(__name__, static_folder='statics', template_folder='templates')


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
  print(request.form)
  print(request.files['myfile'])
  return render_template('upload.html')


if __name__=="__main__":
  app.run(debug=True)