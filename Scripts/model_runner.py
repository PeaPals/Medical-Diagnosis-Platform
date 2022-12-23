# Dependencies

import numpy as np
from PIL import Image, ImageOps

import onnxruntime
import json


# def pneumonia_onnx_model(input_buffer):
#     model_path = "./Models/Pneumonia_Onnx_Model/Pneumonia_Onnx_Model.onnx"
#     diseases = [["Pneumonia", 0.9118]]

#     # preprocessing
#     input_img = Image.open(input_buffer)
#     input_img = ImageOps.grayscale(input_img)
#     input_img = np.array(input_img.resize((500, 500))) / 255

#     while len(input_img.shape) < 4:
#         input_img = np.expand_dims(input_img, axis=0)

#     # input_img = np.expand_dims(input_img, axis=0)
#     # input_img = np.expand_dims(input_img, axis=0)

#     input_img = np.reshape(input_img, (1, 500, 500, 1))

#     # converting data for onnx model
#     data = json.dumps({'data': input_img.tolist()})
#     data = np.array(json.loads(data)['data']).astype('float32')

#     # starting onnx session and model
#     session = onnxruntime.InferenceSession(model_path, None)
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name

#     # running onnx model
#     result = session.run([output_name], {input_name: data})
#     return result[0][0], diseases




# def CheXNet_Onnx_Model(model_path, input_buffer):

#     # preprocessing
#     input_img = Image.open(input_buffer).convert("RGB")
#     input_img = input_img.resize((256, 256))

#     input_img = np.array(input_img.getdata()).reshape(1, 3, input_img.size[0], input_img.size[1]) / 255

#     # converting data for onnx model
#     data = json.dumps({'data': input_img.tolist()})
#     data = np.array(json.loads(data)['data']).astype('float32')

#     # starting onnx session and model
#     session = onnxruntime.InferenceSession(model_path, None)
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name

#     # running onnx model
#     result = session.run([output_name], {input_name: data})
#     return result[0][0]







def run_model(model_path, input_buffer, resize, reshape, image_mode, diseases):
    # preprocessing
    input_img = Image.open(input_buffer).convert(image_mode).resize(resize)
    input_img = np.array(input_img).reshape(reshape) / 255


    # converting data for onnx model
    data = json.dumps({'data': input_img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')

    # starting onnx session and model
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # running onnx model
    result = session.run([output_name], {input_name: data})
    return result[0][0], diseases



def predict(input_buffer, database: dict):
    graph_data = {}

    for model_type, data in database.items():
        predictions, diseases = run_model(input_buffer=input_buffer, **data)
        graph_data[model_type] = (predictions, diseases)

    return graph_data




# CheXNet_Onnx_Model(
#     model_path="../Models/CheXNet_Onnx_Model/CheXNet_Onnx_Model.onnx",
#     input_buffer="../img.png"
# )