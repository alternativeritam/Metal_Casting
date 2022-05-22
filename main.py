import streamlit as st
from PIL import Image
from model import create_model,predict_single
import torchvision.transforms as transforms
import numpy as np
import cv2
from edge import Canny_detector
from region import Kmeans_cluster

def load_image(image_file):
	img = Image.open(image_file)
	return img

def convert_image(img):
    im = Image.open(img).convert('RGB')
    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    opencvImage = cv2.resize(opencvImage,(300,300))
    return opencvImage

transform = transforms.Compose([
    transforms.ToTensor()
])

labels = {
    0: 'Casting is defective',
    1: 'Casting is ok'
}

def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            result.append(labels[i])
    return ' '.join(result)


st.subheader("Please upload a metal casting picture")
image_file = st.file_uploader("Upload Here", type=["png","jpg","jpeg"])

if image_file is not None:

    st.image(load_image(image_file),width=250)

    predict = st.button("Predict the condition")

    if predict:

        status = True
        status = create_model()

        if(status):
            #im_pil = Image.fromarray(im_np)
            img_tensor = transform(load_image(image_file))
            pre = decode_target(predict_single(img_tensor))
            if(pre=="Casting is defective"):
                st.warning(pre)
            else:
                st.success('Casting is ok')
            img = convert_image(image_file)
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
            edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
            cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
            mask = np.zeros((256,256), np.uint8)
            masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
            edge_img = Canny_detector(img)
            region_img = Kmeans_cluster(img,3)
            st.image(region_img,channels="RGB")
            st.image(edge_img,clamp=True,channels="RGB")

        if(status==False):
            st.spinner("Plaease wait we are processing you request")
