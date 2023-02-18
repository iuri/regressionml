import cv2
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import time 

def plot_image(image, figsize = 10, cvtColorParam=None):
    plt.figure(figsize = (figsize, figsize))

    # plt.subplot(figsize[0], figsize[1], i+1)
    if cvtColorParam != None:
        image = cv2.cvtColor(image, cvtColorParam)

    plt.imshow(image)    #If the image is grayscale, as in our case, then we will reshape the output in the following way.
                                                                        #Also, we set the coloring to grayscale so that it doesn't look like it came out of an infrared camera :)
    # else:
    #     plt.imshow(image.reshape((img_rows, img_cols, 1)))
    plt.axis('off')

    # plt.tight_layout()   #Tight layout so that all of the generated images form a nice grid
        
    plt.show()

def get_image_from_webcam():
    # Webcam
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()

    return img
    
def save_image_from_webcam(folder):
    # Webcam
    img = get_image_from_webcam()
    name = f"{folder}/{datetime.datetime.now()}.png"
    cv2.imwrite(name , img)

    return name

def save_image(cv2image, filename=f"assets/{datetime.datetime.now()}.png"):
    sucess = cv2.imwrite(filename , cv2image)
    print(f'Imagem salva em: {filename}')
    return sucess


#Função para retornar tamanho da amostra e passo nos frames
def amostra(n_frames, e):
    import math
    import scipy.stats as st    
    
    Z = st.norm.ppf(1 - e/2) #Z-escore
    P = 0.50 #Desvio padrão 
    C = (Z**2)*P*(1-P)/(e**2) #Constante
    amos = math.ceil(C/(1 + C/n_frames))
    step = math.floor(n_frames/amos)
    return amos, step

def extract_images_from_video(file, error_rate = 0.1):
    import cv2

    video = cv2.VideoCapture(file)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    e = float(error_rate)
    
    amos, step = amostra(n_frames, e)
    
    images = []
    success, frame = video.read()
    start = 0

    while success and start + step <= n_frames:   
        images.append(frame)
        save_image(frame, f'assets/dataset/{start + step}.png')

        #Extrai frames específicos a cada passo
        video.set(cv2.CAP_PROP_POS_FRAMES, (start + step))
        success, frame = video.read()

        start += step

    return images

def get_images_from_folder(folder, cutArray = []):
    # cut=(50:600, 330:950)
    import glob

    filenames = sorted(glob.glob(f'{folder}/*.png'), key=os.path.getmtime)
    images = []

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

    return images, filenames

def label_on_image(image, text, color = (0,0,255)):
    i = image.copy()
    cv2.putText(i, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return i 

def auto_canny(image, sigma=0.33):
    import argparse
	
    # compute the median of the single channel pixel intensities
    v = np.median(image)
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
    # return the edged image
    return edged

def play_images(all_images, rows = 1, columns = 1, sec = 60, time_between_frames = 0.2):
    height = all_images[0][0].shape[0]*rows
    width = all_images[0][0].shape[1]*columns

    n_frames = len(all_images[0])
    fps = n_frames // sec

    print(f'FPS: {fps}', f'height: {height}',f'width: {width}')

    for i in range(n_frames):
        if rows == 1:
            if columns == 1:
                im2show = np.hstack([all_images[0][i]])

            elif columns == 2:
                im2show = np.hstack([all_images[0][i], all_images[1][i]])
            
            elif columns == 3:
                im2show = np.hstack([all_images[0][i], all_images[1][i], all_images[2][i]])
            
            elif columns == 4:
                im2show = np.hstack([all_images[0][i], all_images[1][i], all_images[2][i], all_images[3][i]])

        elif rows == 2:
            if columns == 1:
                im2show = np.vstack([
                    np.hstack([all_images[0][i]]),
                    np.hstack([all_images[1][i]]),
                ])

            elif columns == 2:
                im2show = np.vstack([
                    np.hstack([all_images[0][i], all_images[1][i]]),
                    np.hstack([all_images[2][i], all_images[3][i]])
                ])

            elif columns == 3:
                im2show = np.vstack([
                    np.hstack([all_images[0][i], all_images[1][i], all_images[2][i]]),
                    np.hstack([all_images[3][i], all_images[4][i], all_images[5][i]])
                ])

            elif columns == 4:
                im2show = np.vstack([
                    np.hstack([all_images[0][i], all_images[1][i], all_images[2][i], all_images[3][i]]),
                    np.hstack([all_images[4][i], all_images[5][i], all_images[6][i], all_images[7][i]])
                ])
                
        cv2.imshow("image", im2show)
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break

        time.sleep(time_between_frames)

    print(f'Stopped at frame {i}')
    cv2.destroyAllWindows() 

        