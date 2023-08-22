import base64
from io import BytesIO
import requests
from fastapi import FastAPI , status , HTTPException
from pydantic import BaseModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
from loguru import logger
from PIL import Image as PILImage
import cv2
import numpy as np
import json
import time

fer_detector = pipeline(Tasks.facial_expression_recognition, 'damo/cv_vgg19_facial-expression-recognition_fer')
face_detector = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_ulfd')

app = FastAPI()

class Image(BaseModel):
    image: str

async def resize_image(file_object, coordinates):
    image_np = np.array(file_object)
    # 使用坐标来裁剪图像
    x1, y1, x2, y2 = map(int, coordinates)
    cropped_image = image_np[y1:y2, x1:x2]
    # 计算新的高度，保持原始长宽比
    height = int(cropped_image.shape[0] * (320 / cropped_image.shape[1]))
    # 调整图像大小
    resized_image = cv2.resize(cropped_image, (320, height))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return resized_image

async def json_reorganize(list_values, dictionary):
    emotions_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotions_values = [round(value, 2) for value in dictionary['scores']]

    max_emotion_index = emotions_values.index(max(emotions_values))
    max_emotion = emotions_labels[max_emotion_index]

    emotions_dict = {emotions_labels[i]: emotions_values[i] for i in range(len(emotions_labels))}

    result = [
        {
            "box": [int(list_values[0]), int(list_values[1]), int(list_values[2]), int(list_values[3])],
            "emotions": emotions_dict,
            "max_emotion": max_emotion
        }
    ]

    return result

def download(url):
    data = requests.get(url).content
    # 使用PIL从文件对象读取图像
    image = PILImage.open(BytesIO(data))
    return image

def b64_decode(b64_file):
    # 解码base64数据
    image_data = base64.b64decode(b64_file)
    # 使用PIL从字节数据读取图像
    image = PILImage.open(BytesIO(image_data))
    return image

@app.post("/fer", tags=["FER"], summary="Predict emotions in an image", response_description="Emotion detection results")
async def predict_image(items:Image):
    try:
        pil_data = download(items.image)
        results_face = face_detector(pil_data)
        try:
            points = results_face['boxes'][0]
            resized_image = await resize_image(pil_data, points)
            results_fer = fer_detector(resized_image)
            results_fer = fer_detector(resized_image)
            results_final = await json_reorganize(points, results_fer)
        except:
            results_final = []
        # results[0]['max_emotion'] = max_emotion
        # json_data = json.dumps(results_final)
        # json_data = json_data.replace("'", '"')
        logger.info(results_final)
        return results_final
    except Exception as e:
        errors = str(e)
        mod_errors = errors.replace('"', '**').replace("'", '**')
        logger.error(mod_errors)
        message = {
            "err_no": "400",
            "err_msg": mod_errors
            }
        json_data = json.dumps(message)
        json_data = json_data.replace("'", '"')
        return json_data

@app.post("/fer_b64")
async def b64_predict_image(items:Image):
    try:
        pil_data = b64_decode(items.image)
        results_face = face_detector(pil_data)
        try:
            points = results_face['boxes'][0]
            resized_image = await resize_image(pil_data, points)
            results_fer = fer_detector(resized_image)
            results_fer = fer_detector(resized_image)
            results_final = await json_reorganize(points, results_fer)
        except:
            results_final = []
        # results[0]['max_emotion'] = max_emotion
        # json_data = json.dumps(results_final)
        # json_data = json_data.replace("'", '"')
        logger.info(results_final)
        return results_final
    except Exception as e:
        errors = str(e)
        mod_errors = errors.replace('"', '**').replace("'", '**')
        logger.error(mod_errors)
        message = {
            "err_no": "400",
            "err_msg": mod_errors
            }
        json_data = json.dumps(message)
        json_data = json_data.replace("'", '"')
        return json_data
        
@app.get("/health")
async def health_check():
    try:
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

@app.get("/health/inference")
async def health_check():
    try:
        results = fer_detector("/home/fer/test.jpg")
        logger.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)


