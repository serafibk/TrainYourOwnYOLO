import os
import sys
import cv2
import time
import csv


def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

#image_test_folder = os.path.join(image_folder, "Test_Images")
#image_test_folder = os.path.join(image_folder, "Test_Images_serafina")
image_test_folder = os.path.join(image_folder, "Test")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results/test_result")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

FLAGS = None

def horizontal_line_finder(height, width, pixel_data): #normal finds black lines
    final_out = [] 
    search_dist = 3 
    for y in range(search_dist, height-search_dist):
        short_line = 0
        line_dist = 0
        fails = 0 
        for x in range(width): 
            top = 0
            bot = 0
            for y2 in range(y-search_dist,y-1):
                top += pixel_data[y2,x]/(search_dist-1)

            for y2 in range(y+2,y+search_dist+1):
                bot += pixel_data[y2,x]/(search_dist-1)

            if((top/2+bot/2 - pixel_data[y,x]) > 30): #these are 8 bit ints need to calculate like this to avoid overflow
                line_dist += 1
                if(fails > 0):
                    fails -= 1
            elif(fails < 1): #tolerate x fails
                fails += width/8
            else:
                if(line_dist > width/16):
                    short_line += 1
                line_dist = 0

            if(line_dist > width/8 or short_line >= 4):
                final_out.append(y)  
                break
    return final_out

def vertical_line_finder(height, width, pixel_data, hor_margin_lines): #normal finds black lines
    final_out = [] 
    search_dist = 3
    for x in range(search_dist, width-search_dist):
        line_dist = 0
        fails = 0
        for y in range(height):
            if(y not in hor_margin_lines):
                max_left = 0
                max_right = 0
                for x2 in range(x-search_dist,x):
                    if((pixel_data[y,x2]) > max_left):
                        max_left = pixel_data[y,x2]

                for x2 in range(x+1,x+search_dist+1):
                    if((pixel_data[y,x2]) > max_right):
                        max_right = pixel_data[y,x2]

                if((max_left/2+max_right/2 - pixel_data[y,x]) > 30): #these are 8 bit ints need to calculate like this to avoid overflow
                    line_dist += 1
                    if(fails > 0):
                        fails -= 1
                elif(fails < 1): #tolerate x fails
                    fails += height/8
                else:
                    line_dist = 0 

                if(line_dist > height/8):
                    final_out.append(x)  
                    break      
    return final_out

def real_line_margins(lines, margin_size_pixels):
    margin_lines = []
    for line in lines:
        for i in range(line-margin_size_pixels, line+margin_size_pixels):
            if(i not in margin_lines and i >= lines[0] and i <= lines[-1]):
                margin_lines.append(i)
    return margin_lines

def preprocessing_state(TempImages_dir, TempSlice_locs):
    imgs = os.listdir(image_test_folder)
    imgs.sort()
    img_locs = []
    count_img = 0
    for file in imgs:
        if(count_img != 0):
            img_locs.append(os.path.join(image_test_folder,file))
        count_img += 1
    
    for i in range(len(img_locs)):
        img_loc = img_locs[i]
        print(img_loc)
        pixel_data = cv2.imread(img_loc, 0)
        pixel_data_unchanged = np.copy(pixel_data)

        height, width = pixel_data.shape
        scale = width/800
        pixel_data = cv2.resize(pixel_data,(800, int(height/scale)))
        height, width = pixel_data.shape

        if(0):
            cv2.imshow("image", pixel_data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        hor_time_start = time.time()
        hor_lines = horizontal_line_finder(height, width, pixel_data) 
        hor_margin_lines = real_line_margins(hor_lines, 5)
        hor_time_end = time.time()
        print("--- %s seconds finding horizontal lines---" % (hor_time_end - hor_time_start))
        
        ver_time_start = time.time()
        ver_lines = vertical_line_finder(height, width, pixel_data, hor_margin_lines)
        ver_margin_lines = real_line_margins(ver_lines, 5)
        ver_time_end = time.time()
        print("--- %s seconds finding vertical lines---" % (ver_time_end - ver_time_start))
        """
        if(not hor_lines or not ver_lines):
            return False
        """
        max_x_num = len(ver_lines) - 1
        max_y_num = len(hor_lines) - 1
        min_x = int(ver_lines[0] * scale)
        max_x = int(ver_lines[max_x_num] * scale)
        min_y = int(hor_lines[0] * scale)
        max_y = int(hor_lines[max_y_num] * scale)

        transform_image = pixel_data_unchanged[min_y:max_y, min_x:max_x]
        loc = os.path.join(TempImages_dir, str(img_loc[-23:-4]) + ".jpg") # need to accomodate for your training image name
        cv2.imwrite(loc, transform_image)
        TempSlice_locs[img_loc[-23:-4]] = [min_x, max_x, min_y, max_y]
        #print(TempSlice_locs)
        #cv2.imshow("after_slice", transform_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if(0):
            print(hor_lines)
            print(ver_lines)
            max_x = len(ver_lines) - 1
            max_y = len(hor_lines) - 1
            cv2.line(pixel_data, (ver_lines[0], hor_lines[0]), (ver_lines[max_x], hor_lines[0]), (0,255,0), 4)
            cv2.line(pixel_data, (ver_lines[0], hor_lines[max_y]), (ver_lines[max_x], hor_lines[max_y]), (0,255,0), 4)
            cv2.line(pixel_data, (ver_lines[0], hor_lines[0]), (ver_lines[0], hor_lines[max_y]), (0,255,0), 4)
            cv2.line(pixel_data, (ver_lines[max_x], hor_lines[0]), (ver_lines[max_x], hor_lines[max_y]), (0,255,0), 4)
            cv2.imshow("scaled", pixel_data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if(0):
            for hor_line in hor_lines:
                print(hor_line)
                cv2.line(pixel_data, (0, hor_line), (width, hor_line), (0,255,0), 4)
            for ver_line in ver_lines:
                print(ver_line)
                cv2.line(pixel_data, (ver_line, 0), (ver_line, height), (0,255,0), 4)
            cv2.imshow("scaled", pixel_data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #TempSlice_locs.append(transform_image)
    #return True
    #print(img_locs)


if __name__ == "__main__":
    TempSlice_locs = {}
    TempImages_dir = os.path.join(image_folder, "TempImages")
    try:
        os.makedirs(TempImages_dir)
        print("Directory " , TempImages_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , TempImages_dir ,  " already exists")
        print("Cleaning ipxact directory ...")
        if len(os.listdir(TempImages_dir)) != 0:
            for file in os.listdir(TempImages_dir):
                os.remove(os.path.join(TempImages_dir,file))
    REGISTER = True
    preprocessing_state(TempImages_dir, TempSlice_locs) #sign to determine if registables (not developed yet)
    #print(REGISTER)
    
    
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    if(REGISTER):
        image_reg_folder = os.path.join(image_folder, "TempImages")
        parser.add_argument(
            "--input_path",
            type=str,
            default=image_reg_folder,
            help="Path to image/video directory. All subdirectories will be included. Default is "
            + image_reg_folder,
        )
    else:
        parser.add_argument(
            "--input_path",
            type=str,
            default=image_test_folder,
            help="Path to image/video directory. All subdirectories will be included. Default is "
            + image_test_folder,
        )

    parser.add_argument(
        "--output",
        type=str,
        default=detection_results_folder,
        help="Output path for detection results. Default is "
        + detection_results_folder,
    )

    parser.add_argument(
        "--no_save_img",
        default=False,
        action="store_true",
        help="Only save bounding box coordinates but do not save output images with annotated boxes. Default is False.",
    )

    parser.add_argument(
        "--file_types",
        "--names-list",
        nargs="*",
        default=[],
        help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4",
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        dest="model_path",
        default=model_weights,
        help="Path to pre-trained weight files. Default is " + model_weights,
    )

    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path,
    )

    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        default=model_classes,
        help="Path to YOLO class specifications. Default is " + model_classes,
    )

    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPU to use. Default is 1"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.25,
        help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
    )

    parser.add_argument(
        "--box_file",
        type=str,
        dest="box",
        default=detection_results_file,
        help="File to save bounding box results to. Default is "
        + detection_results_file,
    )

    parser.add_argument(
        "--postfix",
        type=str,
        dest="postfix",
        default="_catface",
        help='Specify the postfix for images with bounding boxes. Default is "_catface"',
    )

    FLAGS = parser.parse_args()

    save_img = not FLAGS.no_save_img

    file_types = FLAGS.file_types

    if file_types:
        input_paths = GetFileList(FLAGS.input_path, endings=file_types)
    else:
        input_paths = GetFileList(FLAGS.input_path)

    # Split images and videos
    img_endings = (".jpg", ".jpg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = FLAGS.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # define YOLO detector
    print(FLAGS.model_path)
    print(FLAGS.classes_path)
    print(FLAGS.score)
    yolo = YOLO(
        **{
            "model_path": FLAGS.model_path,
            "anchors_path": FLAGS.anchors_path,
            "classes_path": FLAGS.classes_path,
            "score": FLAGS.score,
            "gpu_num": FLAGS.gpu_num,
            "model_image_size": (416, 416),
        }
    )

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )

    # labels to draw on images
    class_file = open(FLAGS.classes_path, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

    if input_image_paths:
        print(
            "Found {} input images: {} ...".format(
                len(input_image_paths),
                [os.path.basename(f) for f in input_image_paths[:5]],
            )
        )
        start = timer()
        text_out = ""

        # This is for images
        for i, img_path in enumerate(input_image_paths):
            print(img_path)
            prediction, image = detect_object(
                yolo,
                img_path,
                save_img=save_img,
                save_img_path=FLAGS.output,
                postfix=FLAGS.postfix,
            )
            #print(prediction)
            y_size, x_size, _ = np.array(image).shape
            for single_prediction in prediction:
                out_df = out_df.append(
                    pd.DataFrame(
                        [
                            [
                                os.path.basename(img_path.rstrip("\n")),
                                img_path.rstrip("\n"),
                            ]
                            + single_prediction
                            + [x_size, y_size]
                        ],
                        columns=[
                            "image",
                            "image_path",
                            "xmin",
                            "ymin",
                            "xmax",
                            "ymax",
                            "label",
                            "confidence",
                            "x_size",
                            "y_size",
                        ],
                    )
                )
        end = timer()
        print(
            "Processed {} images in {:.1f}sec - {:.1f}FPS".format(
                len(input_image_paths),
                end - start,
                len(input_image_paths) / (end - start),
            )
        )
        out_df.to_csv(FLAGS.box, index=False)

    # This is for videos
    if input_video_paths:
        print(
            "Found {} input videos: {} ...".format(
                len(input_video_paths),
                [os.path.basename(f) for f in input_video_paths[:5]],
            )
        )
        start = timer()
        for i, vid_path in enumerate(input_video_paths):
            output_path = os.path.join(
                FLAGS.output,
                os.path.basename(vid_path).replace(".", FLAGS.postfix + "."),
            )
            detect_video(yolo, vid_path, output_path=output_path)

        end = timer()
        print(
            "Processed {} videos in {:.1f}sec".format(
                len(input_video_paths), end - start
            )
        )
    # Close the current yolo session
    yolo.close_session()
    
    temp_img_results = os.listdir(detection_results_folder)
    temp_img_results.sort()
    temp_img_locs = []
    TempSlice_img = {}
    for temp in temp_img_results:
        if(temp != "Detection_Results.csv" and temp != ".DS_Store"):
            temp_img_locs.append(os.path.join(detection_results_folder, temp))
    for i in range(len(temp_img_locs)):
        temp_img_loc = temp_img_locs[i]
        #print(temp_img_loc)
        pixel_data = cv2.imread(temp_img_loc)
        key = temp_img_loc[-31:-12] # need to accomodate for training_set
        #print(temp_img_loc[-31:-12])
        TempSlice_img[key] = pixel_data

    test_imgs = os.listdir(image_test_folder)
    test_imgs.sort()
    test_img_locs = []
    for test_img in test_imgs:
        if(test_img != ".DS_Store"):
            test_img_locs.append([os.path.join(image_test_folder,test_img), test_img[:-4]])
            #print(test_img[:-4])
    for i in range(len(test_img_locs)):
        test_img_loc = test_img_locs[i][0]
        key = test_img_locs[i][1]
        original_pixel = cv2.imread(test_img_loc)
        modify_region = TempSlice_locs[key]
        modify_content = TempSlice_img[key]
        #print(modify_region)
        #print(modify_content.shape)
        original_pixel[modify_region[2]:modify_region[3], modify_region[0]:modify_region[1]] = modify_content
        cv2.imwrite(os.path.join(detection_results_folder, key + "_catface.jpg"), original_pixel)
        if(0):
            cv2.imshow("after", original_pixel)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    changed_array = []
    with open(os.path.join(image_folder, "Test_Image_Detection_Results/test_result/Detection_Results.csv"), "r") as csv_register: #can be changed with sign
        rows = csv.reader(csv_register)
        count_row = 0;
        for row in rows:
            if(count_row != 0):
                #print(row[2])
                key = row[0][:-4]
                min_x = int(row[2])
                min_y = int(row[3])
                max_x = int(row[4])
                max_y = int(row[5])
                row[1] = os.path.join(image_test_folder, row[0])
                img = cv2.imread(row[1])
                min_x += TempSlice_locs[key][0]
                max_x += TempSlice_locs[key][0]
                min_y += TempSlice_locs[key][2]
                max_y += TempSlice_locs[key][2]
                row[2] = str(min_x)
                row[3] = str(min_y)
                row[4] = str(max_x)
                row[5] = str(max_y)
                #cv2.line(img, (min_x, min_y), (max_x, min_y), (0,255,0), 4)
                #cv2.line(img, (min_x, max_y), (max_x, max_y), (0,255,0), 4)
                #cv2.line(img, (min_x, min_y), (min_x, max_y), (0,255,0), 4)
                #cv2.line(img, (max_x, min_y), (max_x, max_y), (0,255,0), 4)
                #cv2.imshow("after", img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            changed_array.append(row)
            count_row += 1
    #print(changed_array)
    with open(os.path.join(image_folder, "Test_Image_Detection_Results/test_result/Detection_Results_registers.csv"), "w", newline="") as csv_write:
        writer = csv.writer(csv_write)
        writer.writerows(changed_array)
    os.remove(os.path.join(image_folder, "Test_Image_Detection_Results/test_result/Detection_Results.csv"))
    