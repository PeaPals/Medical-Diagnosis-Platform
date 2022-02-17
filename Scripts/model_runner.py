# Dependencies

import numpy as np
from PIL import Image, ImageOps

import onnxruntime
import json



def pneumonia_onnx_model(model_path, input_buffer):
    
    # preprocessing
    input_img = Image.open(input_buffer)
    input_img = ImageOps.grayscale(input_img)
    input_img = np.array(input_img.resize((500, 500))) / 255

    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.expand_dims(input_img, axis=0)

    input_img = np.reshape(input_img, (1, 500, 500, 1))

    # converting data for onnx model
    data = json.dumps({'data': input_img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')

    # starting onnx session and model
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # running onnx model
    result = session.run([output_name], {input_name: data})
    return result


def CheXNet_Onnx_Model(model_path, input_buffer):
    # preprocessing
    input_img = Image.open(input_buffer).convert("RGB")
    input_img = input_img.resize((256, 256))
    input_img = np.array(input_img.getdata()).reshape(1, 3, input_img.size[0], input_img.size[1]) / 255

    # converting data for onnx model
    data = json.dumps({'data': input_img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')

    # starting onnx session and model
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # running onnx model
    result = session.run([output_name], {input_name: data})
    
    return result[0][0][0], input_buffer


def predict(model, input_buffer, output_box, segmented_image_viewer=None):
    if model == "Pneumonia_Onnx_Model":
        model_prediction = pneumonia_onnx_model(model_path = "Models/" + model + "/" + model + ".onnx" , input_buffer = input_buffer)
        model_prediction = model_prediction[0][0][0]

        output_shower(
            model_prediction = model_prediction,
            disease = "pneumonia",
            output_box = output_box
        )


    
    elif model == "CheXNet_Onnx_Model":
        model_prediction, segmented_image = CheXNet_Onnx_Model(model_path = "Models/" + model + "/" + model + ".onnx" , input_buffer = input_buffer)
        
        output_shower(
            model_prediction = model_prediction,
            disease = "pneumonia",
            output_box = output_box
        )

        segmented_image_viewer.image(segmented_image, caption="disease diagnosis prediction chart")



def output_shower(model_prediction, disease, output_box):
    if model_prediction >= 0.5:
        confidence = round(model_prediction*100, 2)
        output_text = f"This seems to be the case of {disease}... I am {confidence}% sure"
        output_box.error(output_text)

    else:
        confidence = round((1-model_prediction)*100, 2)
        output_text = f"This seems to be a normal case ... I am {confidence}% sure"
        output_box.success(output_text)