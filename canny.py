# -*- coding: utf-8 -*-
"""Canny ok (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CePE7AeOMYyCE6Yw4cmNRMuFqJKoozcz
"""
import os
import cv2
import glob
import utils
import numpy as np
import time
import re

folder = "camera1/samples"

# images, files = utils.get_images_from_folder(folder, cutArray=[50,600,330,950])


filenames = sorted(glob.glob(f'{folder}/*.png'), key=os.path.getmtime)
images = []
cutArray = []
for file in filenames:
    img = cv2.imread(file)
    if len(cutArray) > 0:
        images.append(
            img[
                cutArray[0] : cutArray[1], 
                cutArray[2] : cutArray[3]
            ]
        )

    else:
        images.append(img)


f'Foram carregadas {len(images)} imagens'

sec = 60
rows = 2
columns = 3
height = images[0].shape[0]*rows
width = images[0].shape[1]*columns

n_frames = len(images)
fps = n_frames // sec

f'FPS: {fps}', f'height: {height}',f'width: {width}'

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter("outputs/canny.mp4", fourcc, float(fps), (width, height))
a = 75
b = 175    
try:
    for image in images:
        #print("NPARRAY",image)
        #print("NPARRAY", type(image))
        txt = type(image)
        if image is not None:
            #print("yes")
            if image.any():
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frame = cv2.GaussianBlur(frame,(7,7), 0)
                
                # Canny from image
                suavizacao = cv2.GaussianBlur(frame,(7,7), 0)
                canny_from_frame = cv2.Canny(suavizacao, a, b) #, L2gradient = True)

                # canny_from_frame = cv2.bitwise_and(frame, frame, mask=canny_from_frame)
                canny_from_frame = cv2.cvtColor(canny_from_frame, cv2.COLOR_GRAY2BGR)

                # threshold
                thresh = cv2.threshold(frame, 60, 255, cv2.THRESH_BINARY)[1]

                # morphology edgeout = dilated_mask - mask
                # morphology dilate
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

                # get absolute difference between dilate and thresh
                diff = cv2.absdiff(dilate, thresh)

                # invert
                edges = 255 - diff
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                ret, img_otsu = cv2.threshold(suavizacao, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                canny_from_otsu = cv2.Canny(img_otsu, a, b)
                canny_from_otsu = cv2.cvtColor(canny_from_otsu, cv2.COLOR_GRAY2BGR)

                autoCannyOtsu = utils.auto_canny(img_otsu)
                autoCannyOtsu = cv2.cvtColor(autoCannyOtsu, cv2.COLOR_GRAY2BGR)
                
                autoCanny = utils.auto_canny(frame)
                autoCanny = cv2.cvtColor(autoCanny, cv2.COLOR_GRAY2BGR)
                
                img_otsu = cv2.bitwise_not(img_otsu)
                img_otsu = cv2.cvtColor(img_otsu, cv2.COLOR_GRAY2BGR)

                # Labels on images
                image = utils.label_on_image(image, "Real")
                img_otsu = utils.label_on_image(img_otsu, "Threshold OTSU")
                canny_from_frame = utils.label_on_image(canny_from_frame, "Canny")

                # autoCanny2 = utils.label_on_image(autoCanny2, "Auto canny 0.66")
                autoCanny = utils.label_on_image(autoCanny, "Auto canny")
                edges = utils.label_on_image(edges, "edges")
                canny_from_otsu = utils.label_on_image(canny_from_otsu, "Canny from OTSU")
                autoCannyOtsu = utils.label_on_image(autoCannyOtsu, "Autocanny from otsu")
                im2show = np.vstack([
                    np.hstack([image, canny_from_frame, autoCanny]),
                    np.hstack([img_otsu, canny_from_otsu, autoCannyOtsu])
                    # np.hstack([canny_from_frame, sobel, laplacian])
                ])
                cv2.imshow("image", im2show)
                video.write(im2show)
                k = cv2.waitKey(30) & 0xff
                if k == 27: 
                    break
                time.sleep(0.2)
except Exception as e:
    print("Error ", e)
    pass
cv2.destroyAllWindows() 
    
video.release()
