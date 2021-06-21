import cv2,os
from flask import Flask, render_template,request
from keras.models import load_model
from keras.models import load_model
import numpy as np
appl = Flask(__name__)

model_path='C:\\Users\\Admin\\disease_prediction.h5'
model=load_model(model_path)
diseases_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
labels=['nv','mel','bkl','bcc','akiec','vasc','df']
@appl.route('/')
def home1():
  return render_template('home1.html')

@appl.route('/result',methods=["POST"])
def result():
  img=request.files['img']
  img.save('C:\\Users\\Admin\\project_bhavya\\static\\images\\img.jpg')
  img=cv2.imread('C:\\Users\\Admin\\project_bhavya\\static\\images\\img.jpg')
  img=cv2.resize(img,(32,32))
  img=img.reshape(-1,32,32,3)
  p=model.predict([img])
  x=np.argmax(p)
  return render_template("result.html",data=diseases_dict[labels[x]])

if __name__ == '__main__':
  app.run(debug=True)
