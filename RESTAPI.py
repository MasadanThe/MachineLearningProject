from joblib import load
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd

api = Flask(__name__)

ML = load(Path(__file__).with_name('SupportVector.joblib'))

#Example post
#http://localhost:5000/?bwt=128&gestation=279&parity=0&age=28&height=64&weight=115 
@api.route('/', methods=['POST'])
def get_prediction():
    info = []
    info.append(request.args.get('bwt'))
    info.append(request.args.get('gestation'))
    info.append(request.args.get('parity'))
    info.append(request.args.get('age'))
    info.append(request.args.get('height'))
    info.append(request.args.get('weight'))
    
    prediction = ML.predict([info])
    
    if(prediction[0] == 0):
        return jsonify(answer = "Does not smoke")
    return jsonify(answer = "Do smoke")


api.run()