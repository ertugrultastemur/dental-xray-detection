# -*- coding: utf-8 -*-
# @date: 6.05.2024
# @author: ikbal
# @file: Prediction.py

import cv2
import numpy as np
from PIL import Image
import io
import base64

import log4p

loggerApp = log4p.GetLogger("App", config='Config/log4p.json')
logger = loggerApp.logger

def predict_image(base64_image, model):
    # decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))

    # detect
    results = model(image)
    logger.info("Image detected successfully")

    # output detected image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return base64_image





