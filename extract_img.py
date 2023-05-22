from PIL import Image
import numpy as np
import requests
from io import BytesIO
import sys
import pandas as pd

def extract_img(img_link):
    resp = requests.get(img_link)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    img_arr = np.array(img)
    print(img_arr.shape)
    return img_arr

df = pd.read_csv('newdata.csv')


new_classes = []
new_imgs = []
new_sents = []
for i, r in df.iterrows():  
    new_classes.append(r['Class'])
    path = 'img_arrs/img_arr' + str(i) + '.npy'
    np.save(path, extract_img(r['Image URL']))
    new_imgs.append(path)
    new_sents.append(r['Sentence'])

new_df = pd.DataFrame({'Class': new_classes, 'Image Array': new_imgs, 'Sentence': new_sents})
new_df.to_csv('data.csv')