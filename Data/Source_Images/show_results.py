import cv2
import os
import matplotlib.pyplot as plt
import csv
import xml.etree.ElementTree as ET


root_path = r"/Users/serafinakamp/Desktop/YOLO_test/TrainYourOwnYOLO/Data/Source_Images"

root_xml = r"/Users/serafinakamp/Desktop/TableExt/datasheet-scrubber/src/Table_Extraction_Weight_Creation"


image_folder = "Test_Image_Detection_Results/1000_result"
xml_folder = "xml_bound_train"


def calc_IoU(xml,proposed):
    intersection = 0
    xmlMinX = xml[0]
    xmlMinY = xml[2]
    xmlMaxX = xml[1]
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


images = os.listdir(os.path.join(root_path,image_folder))
images.sort()


num_tables_prop = 0
correct_tables = 0

#to read csv
with open(os.path.join(root_path,"Test_Image_Detection_Results/1000_result/Detection_Results.csv"),'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        num = row[0][16:-4]
        if num: #exclude header row
            proposed=[int(row[2]),int(row[3]),int(row[4]),int(row[5])]#minX minY maxX maxY
            top_left = (proposed[0],proposed[1])
            top_right = (proposed[2],proposed[1])
            bot_left = (proposed[0],proposed[3])
            bot_right = (proposed[2],proposed[3])

            confidence = float(row[7])

            ##get corresponding xml to calculate IoU score
            xml_name = "train_bound_xml_"+num+".xml"

            maxIOU = 0 #find max of all tables in xml compared to proposed

            #read in bounds
            with open(os.path.join(root_xml,xml_folder,xml_name),'r') as xml:
                tree = ET.parse(xml)

                root = tree.getroot()
                for table in root:
                    points = table[0].attrib["points"]

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
                    xml_locs = [min_vals[0],max_vals[0],min_vals[1],max_vals[1]]#minX maxX minY maxY
                    iou = calc_IoU(xml_locs,proposed)
                    if iou > maxIOU:
                        maxIOU = iou
            im_name = "train_bound_img_"+num+"_catface.jpg"
            image = cv2.imread(os.path.join(root_path,image_folder,im_name),0)
            cv2.line(image, top_left, bot_left, (0,255,0), 3)
            cv2.line(image, top_left, top_right, (0,255,0), 3)
            cv2.line(image, bot_left, bot_right, (0,255,0), 3)
            cv2.line(image, top_right, bot_right, (0,255,0), 3)
            image = cv2.resize(image,(600,800))
            cv2.imshow("image",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #if maxIOU > threshold, correctly identifies table
            if maxIOU > 0.6:
                im_name = "train_bound_img_"+num+"_catface.jpg"
                image = cv2.imread(os.path.join(root_path,image_folder,im_name),0)
                '''
                cv2.line(image, top_left, bot_left, (0,255,0), 3)
                cv2.line(image, top_left, top_right, (0,255,0), 3)
                cv2.line(image, bot_left, bot_right, (0,255,0), 3)
                cv2.line(image, top_right, bot_right, (0,255,0), 3)
                image = cv2.resize(image,(600,800))
                '''
                if maxIOU <= 0.7:
                    '''
                    cv2.imshow("image",image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''
                correct_tables+=1
            num_tables_prop+=1

#count ground truth tables
num_tables_xml=0
for img in images[1:]:
    num = img[16:-12]
    xml = "train_bound_xml_"+num+".xml"
    with open(os.path.join(root_xml,xml_folder,xml),'r') as xml:
        tree = ET.parse(xml)

        root = tree.getroot()
        for table in root:
            num_tables_xml=num_tables_xml+1

print("NUM CORRECT: ", correct_tables)
print("TOTAL PROPOSED: ", num_tables_prop)
print("TOTAL TRUE: ", num_tables_xml)
print("PRECISION :", correct_tables/num_tables_prop)
print("RECALL: ", correct_tables/num_tables_xml)
