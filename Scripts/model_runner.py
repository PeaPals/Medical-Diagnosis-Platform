# Dependencies

import numpy as np
from PIL import Image, ImageOps

import onnxruntime
import json



CheXNet_Model_Index_Mapping = {
    "Atelectasis Prediction (from X-Ray)" : 0,
    "Cardiomegaly Prediction (from X-Ray)" : 1,
    "Effusion Prediction (from X-Ray)" : 2,
    "Infiltration Prediction (from X-Ray)" : 3,
    "Mass Prediction (from X-Ray)" : 4,
    "Nodule Prediction (from X-Ray)" : 5,
    "Pneumonia Prediction (from X-Ray) (Using CheXNet)" : 6,
    "Pneumothorax Prediction (from X-Ray)" : 7,
    "Consolidation Prediction (from X-Ray)" : 8,
    "Edema Prediction (from X-Ray)" : 9,
    "Emphysema Prediction (from X-Ray)" : 10,
    "Fibrosis Prediction (from X-Ray)" : 11,
    "Pleural Thickening Prediction (from X-Ray)" : 12,
    "Hernia Prediction (from X-Ray)" : 13 

}


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
    return result[0][0][0]


def CheXNet_Onnx_Model(model_path, disease, input_buffer):
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

    index = CheXNet_Model_Index_Mapping[disease]
    return result[0][0][index]#, input_buffer


def Tbnet_Onnx_Model(model_path, input_buffer):
    # preprocessing
    input_img = Image.open(input_buffer)
    input_img = ImageOps.grayscale(input_img)
    input_img = np.array(input_img.resize((224, 224))) / 255

    input_img = np.expand_dims(input_img, axis=0)
    input_img = np.expand_dims(input_img, axis=0)

    input_img = np.reshape(input_img, (1, 224, 224, 1))

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


def predict(model, disease, input_buffer, output_box, segmented_image_viewer=None):
    if model == "Pneumonia_Onnx_Model":
        model_prediction = pneumonia_onnx_model(model_path = "Models/" + model + "/" + model + ".onnx" , input_buffer = input_buffer)

        output_shower(
            model_prediction = model_prediction,
            disease = disease.split("Prediction")[0],
            output_box = output_box
        )
    
    elif model == "CheXNet_Onnx_Model":
        #model_prediction, segmented_image = CheXNet_Onnx_Model(model_path = "Models/" + model + "/" + model + ".onnx" , disease = disease, input_buffer = input_buffer)
        model_prediction = CheXNet_Onnx_Model(model_path = "Models/" + model + "/" + model + ".onnx", disease = disease, input_buffer = input_buffer)
        
        output_shower(
            model_prediction = model_prediction,
            disease = disease.split("Prediction")[0],
            output_box = output_box
        )


    elif model == "Tbnet_Onnx_Model":
        model_prediction = Tbnet_Onnx_Model(model_path = "Models/" + model + "/" + model + ".onnx", input_buffer = input_buffer)

        output_shower(
            model_prediction = model_prediction,
            disease = disease.split("Prediction")[0],
            output_box = output_box
        )

    # uncomment following line only after adding this model to "segmented_output_models" in database.json file
    #segmented_image_viewer.image(segmented_image, caption="disease diagnosis prediction chart")


def output_shower(model_prediction, disease, output_box):
    if model_prediction >= 0.5:
        confidence = round(model_prediction*100, 2)
        output_text = f"This seems to be the case of {disease}... I am {confidence}% sure"
        output_box.error(output_text)

    else:
        confidence = round((1-model_prediction)*100, 2)
        output_text = f"This seems to be a normal case ... I am {confidence}% sure"
        output_box.success(output_text)