import os
import time
import keras
import cv2
import numpy as np
from keras.models import load_model
import argparse
import csv
import xml.etree.ElementTree as ET


def get_coordinates(points):
    min_vals = [10000, 10000]
    max_vals = [0,0]

    on_X = True
    temp = ""
    for char in points:
        if(char == "," or char == " "):
            if(int(temp) < min_vals[not on_X]):
                min_vals[not on_X] = int(temp)
            if(int(temp) > max_vals[not on_X]):
                max_vals[not on_X] = int(temp)
            on_X = not on_X
            temp = ""
        else:
            temp += char
    return [min_vals[0],min_vals[1],max_vals[0],max_vals[1]]#minX minY maxX, maxY

##calculate the IoU between two regions
def calc_IoU(xml,proposed):
    intersection = 0
    xmlMinX = xml[0]
    xmlMinY = xml[1]
    xmlMaxX = xml[2]
    xmlMaxY = xml[3]
    propMinX = proposed[0]
    propMinY = proposed[1]
    propMaxX = proposed[2]
    propMaxY = proposed[3]
    width_shared = min(propMaxX,xmlMaxX) - max(propMinX,xmlMinX)
    height_shared = min(propMaxY,xmlMaxY) - max(propMinY,xmlMinY)
    if width_shared > 0 and height_shared > 0:
        intersection = width_shared*height_shared
    xmlArea = (xmlMaxX-xmlMinX)*(xmlMaxY-xmlMinY)
    propArea = (propMaxX-propMinX)*(propMaxY-propMinY)
    union = xmlArea + propArea - intersection
    return intersection/union

#detecting tables using current cnns
def cnn_detect(model1,model2,i):
    X_size = 800 #part1
    Y_size = 64 #part1

    pTwo_size = 600 #part2
    cuts_labels = 60 #part2
    label_precision = 8 #AMOUNT OF PIXELS BETWEEN LABELS, GOES FROM 1/4th to 3/4ths

    y_fail_num = 2

    pixel_data = cv2.imread(i, 0)
    original_pixel_data_255 = pixel_data.copy()
    pixel_data = cv2.normalize(pixel_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    original_pixel_data = pixel_data.copy()

    height, width = pixel_data.shape
    scale = X_size/width

    pixel_data = cv2.resize(pixel_data, (X_size, int(height*scale))) #X, then Y
    bordered_pixel_data = cv2.copyMakeBorder(pixel_data,top=int(Y_size/4),bottom=int(Y_size/4),left=0,right=0,borderType=cv2.BORDER_CONSTANT,value=1)

    slice_skip_size = int(Y_size/2)
    iter = 0
    slices = []
    while((iter*slice_skip_size + Y_size) < int(height*scale+Y_size/2)):
        s_iter = iter*slice_skip_size
        slices.append(bordered_pixel_data[int(s_iter):int(s_iter+Y_size)])
        iter += 1

    slices = np.array(np.expand_dims(slices,  axis = -1))

    data = model1.predict(slices)

    conc_data = []
    for single_array in data:
        for single_data in single_array:
            conc_data.append(single_data)
    conc_data += [0 for i in range(y_fail_num+1)] #Still needed
    groups = []
    fail = y_fail_num
    group_start = 1 #start at 1 to prevent numbers below zero in groups
    for iter in range(len(conc_data)-1):
        if(conc_data[iter] < .5):
            fail += 1
        else:
            fail = 0

        if(fail >= y_fail_num):
            if(iter - group_start >= 4):
                groups.append((max(int((group_start-1)*label_precision/scale),0), int((iter+1-y_fail_num)*label_precision/scale)))
            group_start = iter



    groups2 = []
    for group in groups:
        temp_final_original = cv2.resize(original_pixel_data[group[0]:group[1]], (pTwo_size, pTwo_size))
        temp_final = np.expand_dims(np.expand_dims(temp_final_original,  axis = 0), axis = -1)
        data_final = model2.predict(temp_final)

        hor_start = -1
        hor_finish = 10000
        pointless, original_width = original_pixel_data.shape

        for iter in range(len(data_final[0])):
            if(data_final[0][iter] > .5 and hor_start == -1):
                if(iter > 0):
                    hor_start = int((iter-0.5)*original_width/cuts_labels)
                else:
                    hor_start = int(iter*original_width/cuts_labels)

            if(data_final[0][iter] > .5):
                hor_finish = int((iter+0.5)*original_width/cuts_labels)

        if(1 and hor_finish - hor_start > (0.7 * original_width)): #Fix for tables that cover the entire image
            groups2.append((0, original_width))
        else:
            groups2.append((hor_start, hor_finish))


    return groups, groups2 #returns all detected y vals,x vals


#post processing for YOLO detected images - default delta to expand boxes by is 5 px
def post_process_yolo(yolo_csv,delta=5):

    #dictionary to track tables propsed for each image key = [b,f]NUM, ex b37 for bound_37
    tables_on_page = {}

    #to read csv
    with open(yolo_csv,'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "image":
                continue
            if row[0][6] == 'b':
                num = 'b'+row[0][16:-4]
                im_name = "train_bound_img_"+num[1:]+".jpg"
            else:
                num = 'f'+row[0][15:-4]
                im_name = "train_full_img_"+num[1:]+".jpg"
            if num: #exclude header row
                page_width = int(row[8])
                page_height = int(row[9])
                proposed=[max(int(row[2])-delta,0),max(int(row[3])-delta,0),min(int(row[4])+delta,page_width),min(int(row[5])+delta,page_height)]#minX minY maxX maxY
                top_left = (proposed[0],proposed[1])
                bot_right = (proposed[2],proposed[3])

                confidence = float(row[7])

                if num in tables_on_page:
                    max_iou = 0
                    prop_overlap=[]
                    found_ind = 0
                    for i,prop in enumerate(tables_on_page[num]):

                        iou = calc_IoU(prop[0],proposed)
                        if iou>max_iou:
                            max_iou = iou
                            prop_overlap = prop[0]
                            found_ind = i
                    if max_iou < 0.1: #doesn't overlap with already proposed tables
                        tables_on_page[num].append([proposed,confidence])

                    elif prop[1] < confidence: #confidence is higher, so delete previous table
                        tables_on_page[num].append([proposed,confidence])
                        del tables_on_page[num][found_ind]
                        #print("new table is more confident")
                    else:
                        print("overlap and less confident")

                else: #add new key
                    tables_on_page[num] = [[proposed,confidence]]

    return tables_on_page #return dict containing processed tables for each image




if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(description="Detect tables on an image using YOLO and TableExt CNNs")
    """
    Command line options
    """

    #CNN model paths
    parser.add_argument("-c", "--cnn_models",type=str,help="path to cnn models, default is current folder",default=os.getcwd())

    #YOLO detected image path
    parser.add_argument("-y", "--yolo_detected",type=str,help="path to yolo detected tables csv file, default is current folder/Detection_Results.csv",default=os.path.join(os.getcwd(),"Detection_Results.csv"))

    #test image path
    parser.add_argument("-t", "--test_images",type=str,help="path to test images, default is current folder/test_images",default=os.path.join(os.getcwd(),"test_images"))

    #corresponding xml path
    parser.add_argument("-x", "--test_xmls",type=str,help="path to corresponding test xmls, default is current folder/test_xmls",default=os.path.join(os.getcwd(),"test_xmls"))

    #IoU threshold TODO find out what default should be and if this is necessary
    parser.add_argument("-i", "--iou_thresh",type=float,help="threshold for IoU of proposed boxes, default is 0.5",default=0.5)

    FLAGS = parser.parse_args() #to parse arguments

    #define necessary paths
    cnn_models_path = FLAGS.cnn_models
    yolo_csv = FLAGS.yolo_detected
    test_image_path = FLAGS.test_images
    test_xml_path = FLAGS.test_xmls

    #iou Threshold
    IOU_threshold = FLAGS.iou_thresh

    #read in cnn model weights
    model1 = load_model(os.path.join(cnn_models_path, "stage1.h5"))
    model2 = load_model(os.path.join(cnn_models_path, "stage2.h5"))

    #process yolo images
    print("Processesing Yolo detected tables")
    processed_tables = post_process_yolo(yolo_csv)

    #get image_locs
    imgs = os.listdir(test_image_path)
    img_loc=[]
    for im in imgs:
        img_loc.append(os.path.join(test_image_path,im))


    proposed_tables = 0
    correct_tables = 0
    for i_num, i in enumerate(img_loc):
        print("Detecting ", i)
        if i[92] == 'b':
            num = 'b'+i[102:-4]
            xml_filename = "train_bound_xml_"+num[1:]+".xml"
            xml_folder ="xml_bound_train"
        else:
            num = 'f'+i[101:-4]
            xml_filename = "train_full_xml_"+num[1:]+".xml"
            xml_folder ="xml_full_train"
        all_y,all_x = cnn_detect(model1,model2,i)
        if num in processed_tables:
            yolo_tables = processed_tables[num]
        else: #yolo detected no tables
            yolo_tables = []

        img = cv2.imread(i)
        height = img.shape[0]
        width = img.shape[1]
        '''
        ##draw rectangles
        #cnn
        for i in range(len(all_y)):
            top_left = (all_x[i][0],all_y[i][0])
            bot_right = (all_x[i][1],all_y[i][1])

            cv2.rectangle(img, top_left, bot_right,(150,0,150),3)
        ##yolo
        for table in yolo_tables:
            top_left = (table[0][0],table[0][1])
            bot_right = (table[0][2],table[0][3])

            cv2.rectangle(img, top_left, bot_right,(150,150,0),3)
        '''

        final_tables=[]
        #prune
        for table in yolo_tables:
            maxiou = 0
            cnn_found=[]
            yolo_coords=table[0]

            for i in range(len(all_y)):
                cnn_coords = [all_x[i][0],all_y[i][0],all_x[i][1],all_y[i][1]]
                top_left_cnn = (cnn_coords[0],cnn_coords[1])
                bot_right_cnn = (cnn_coords[2],cnn_coords[3])

                top_left_yolo = (yolo_coords[0],yolo_coords[1])
                bot_right_yolo = (yolo_coords[2],yolo_coords[3])

                iou = calc_IoU(cnn_coords,yolo_coords)
                if iou > maxiou:
                    maxiou = iou
                    cnn_found=cnn_coords

                #show IoU
                if 0:
                    cv2.rectangle(img, top_left_cnn, bot_right_cnn,(0,255,0),3)
                    cv2.rectangle(img, top_left_yolo, bot_right_yolo,(0,255,0),3)

                    img = cv2.resize(img,(600,800))
                    cv2.imshow('image',img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    img = cv2.resize(img,(width,height))

                    cv2.rectangle(img, top_left_cnn, bot_right_cnn,(150,0,150),3)
                    cv2.rectangle(img, top_left_yolo, bot_right_yolo,(150,150,0),3)

            if maxiou >=0.76:
                final_min_x = min(cnn_found[0],yolo_coords[0])
                final_min_y = min(cnn_found[1],yolo_coords[1])
                final_max_x = max(cnn_found[2],yolo_coords[2])
                final_max_y = max(cnn_found[3],yolo_coords[3])

                final_tables.append([final_min_x,final_min_y,final_max_x,final_max_y])
            else:#prefer yolo if no overlap
                final_tables.append(yolo_coords)
        #if no yolo tables take all cnn tables
        if len(yolo_tables) == 0:
            for i in range(len(all_y)):
                cnn_coords = [all_x[i][0],all_y[i][0],all_x[i][1],all_y[i][1]]
                final_tables.append(cnn_coords)

                '''
                cv2.rectangle(img, (cnn_coords[0],cnn_coords[1]), (cnn_coords[2],cnn_coords[3]),(0,255,0),3)

                img = cv2.resize(img,(600,800))
                cv2.imshow('image',img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                img = cv2.resize(img,(width,height))
                '''


        for table in final_tables: #draw final rectangles
            #top_left = (table[0],table[1])
            #bot_right = (table[2],table[3])


            proposed_tables+=1
            maxIOU = 0
            with open(os.path.join(test_xml_path,xml_folder,xml_filename),'r') as xml:

                tree = ET.parse(xml)
                root = tree.getroot()

                for child in root:
                    points = child[0].attrib["points"]
                    xml_locs = get_coordinates(points)

                    iou = calc_IoU(xml_locs,table)
                    if iou > maxIOU:
                        maxIOU = iou
            if maxIOU > 0.7:
                correct_tables+=1



###uncomment lines below to see detection results

            #cv2.rectangle(img,top_left,bot_right,(255,0,0),3)

        '''
        img=cv2.resize(img, (600,800))
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

    #count True
    #count ground truth tables
    num_tables_xml=0
    for img in img_loc:
        if img[92] == 'b': #if bound
            num = img[102:-4]
            xml_filename = "train_bound_xml_"+num+".xml"
            xml_folder ="xml_bound_train"
        else: #if full
            num = img[101:-4]
            xml_filename = "train_full_xml_"+num+".xml"
            xml_folder ="xml_full_train"


        with open(os.path.join(test_xml_path,xml_folder,xml_filename),'r') as xml:
            tree = ET.parse(xml)
            root = tree.getroot()
            for table in root:
                num_tables_xml=num_tables_xml+1

    print("PROPOSED TABLES: ", proposed_tables)
    print("CORRECT TABLES: ", correct_tables)
    print("TRUE TABLES: ",num_tables_xml)

    print("PRECISION: ", correct_tables/proposed_tables)
    print("RECALL: ", correct_tables/num_tables_xml)
