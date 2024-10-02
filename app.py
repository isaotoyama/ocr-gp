import easyocr
import cv2
import matplotlib.pyplot as plt
import requests
import torch
import pytesseract
from PIL import Image
import json
import numpy as np

img_path1 = "ReceiptData/20231016_180324.jpg"
img_path2 = "ReceiptData/20231010_210904.jpg"
img_path3 = "ReceiptData/20231014_182753.jpg"
img_path4 = "ReceiptData/20230917_131726.jpg"
img_path5 = "ReceiptData/20231002_190427.jpg"

PATH_TO_USE = img_path2

img = cv2.imread(PATH_TO_USE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = reader.readtext(img, detail=0)

result_string = ""
for ele in result:
 result_string += ele + " "
 
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

OPEN_AI_API_KEY = "123123123"

from constants import OPEN_AI_API_KEY
OPEN_AI_API_KEY = str(OPEN_AI_API_KEY)
assert OPEN_AI_API_KEY.startswith("sk-") and OPEN_AI_API_KEY.endswith("123")
client = OpenAI(api_key=OPEN_AI_API_KEY)


from openai import OpenAI
import os
client = OpenAI(api_key=OPEN_AI_API_KEY)

MODEL = "gpt-4o-mini"

def prompt_gpt(prompt):
 return client.chat.completions.create(
 model=MODEL,
 messages=[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": prompt}
 ]
 ).choices[0].message.content
 
prompt = f"Given a string of text from an OCR of a receipt. Find each item and price in the receipt, and return with a list of tuples like this: [(item1, price1), (item2, price2), ...]. Only respond with the list, and nothing else. The string is: {result_string}"
prompt += " . Sure, here is the list of items and prices: " 

response = prompt_gpt(prompt)
 
 # for image 'ReceiptData/20231010_210904.jpg'
[("KylLING HotwiNGS", "57,00"), ("Iskaffe Hocca", "18,90"), ("TORTILLACHIP ZINGY", "16,90"), ("SøTPOTeT FRIES", "37,00"), ("Creamy PEANøTTSHeR", "46,00"), ("GluTEn FReE TORT", "43,90"), ("DIP TEX MEX STyLE", "40,90")]

# for image 'ReceiptData/20231016_180324.jpg'
[('RISTO HOZZA _ 2PK', 89.90), ('SUPERHEL T , GROYBRøP', 35.00), ('B#REPOSE', 25.00), ('Dr Oetker', 26.97)]

# for image 'ReceiptData/20231002_190427.jpg'
[('TøRKEDE APRIKOSER', 29.90), ('MANDLER', 10.90), ('Couscous', 22.40), ('FISKEBURGER HYS8TO', 53.90), ('AVOCADO 2PK', 40.00), ('GRøNNKÅL', 0.00), ('BROKKOLI', 0.00), ('GULROT BEGER 75OGR', 3.00)]

result2 = reader.readtext(img, detail=1)

# Loop through the results and draw bounding boxes on the original image
for (bbox, text, prob) in result2:
    top_left = tuple(bbox[0])
    bottom_right = tuple(bbox[2])
    
    # Draw the bounding box on the original image
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Green box with thickness 2
    
    # Optionally, put the recognized text on the original image
    cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Now resize the image to a smaller sizescale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize the image
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Save or display the resized image with bounding boxes
cv2.imwrite('output_image_with_boxes.jpg', resized_img)
cv2.imshow('Resized Image with Bounding Boxes', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import boto3
from io import BytesIO
from constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
import os

def get_aws_textract_client():
 return boto3.client('textract',
      aws_access_key_id=AWS_ACCESS_KEY_ID,
      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
      region_name=AWS_REGION)

def get_textract_text_from_image(client, image_path):
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    with open(image_path, 'rb') as document:
        img = bytearray(document.read())

    # Call Amazon Textract
    response = client.detect_document_text(
        Document={'Bytes': img}
    )
    return response

def extract_text_from_response(response):
    result_string = ""
    for block in response["Blocks"]:
        if block["BlockType"] == "WORD" or block["BlockType"] == "LINE":
            result_string += block["Text"] + " "
    return result_string

[('TORKEDE APRIKOSER', 29.90), ('MANDLER', 10.90), ('COUSCOUS', 22.40), ('FISKEBURGER HYS&TO', 53.90), ('AVOCADO 2PK 320G', 34.90), ('GRONNKAL 150G', 24.90), ('BROKKOLI', 24.90), ('GULROT BEGER 750GR', 24.90)]
[('RISTO. MOZZA. 2PK 15%', 89.90), ('SUPERHELT GROVBROD 15%', 35.00), ('BAREPOSE 80% RESIR 25%', 4.25), ('30% Dr. Oetker', -26.97)]


