# -*- coding: utf-8 -*-
# @date: 6.05.2024
# @author: ikbal
# @file: Service.py


import ultralytics, log4p
from Prediction import predict_image
from flask import Flask, request, jsonify
from Utils import get_json_config


# Initialize logger
loggerApp = log4p.GetLogger("App", config='Config/log4p.json')
logger = loggerApp.logger

app = Flask(__name__)

@app.route('/detect_dental_image', methods=['POST'])
def predict():
    # get base64 image
    data = request.json

    image = data['image']

    # predict
    base64_image = predict_image(image, model)


    return jsonify({'image': base64_image})


if __name__ == '__main__':
    model = ultralytics.YOLO('model/best.pt')
    logger.info("Model loaded successfully")
    service_configs = get_json_config('Config/serviceConfig.json')
    app.run(host=service_configs['host'], port=service_configs['port'])
