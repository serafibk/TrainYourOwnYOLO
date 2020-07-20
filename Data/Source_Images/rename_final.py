import os
import cv2

root = r"/Users/serafinakamp/Desktop/YOLO_test/TrainYourOwnYOLO/Data/Source_Images"

image_final_folder = "Test_Image_Detection_Results"

images = os.listdir(os.path.join(root,image_final_folder))
images.sort()

for im in images[1:]:
    num = im[16:-12]
    im_name = "detection_result_100_"+num+".jpg"
    os.rename(os.path.join(image_final_folder,im),os.path.join(image_final_folder,im_name))
