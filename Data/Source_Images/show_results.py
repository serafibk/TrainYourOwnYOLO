import cv2
import os
import matplotlib.pyplot as plt
import csv
import xml.etree.ElementTree as ET

##change these to match local directory path
root_path = r"/Users/serafinakamp/Desktop/YOLO_test/TrainYourOwnYOLO/Data/Source_Images"

root_xml = r"/Users/serafinakamp/Desktop/TableExt/datasheet-scrubber/src/Table_Extraction_Weight_Creation"

image_folder = "Test_Image_Detection_Results/1700_result"


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

##from table_identification_final_2.py to get coordinates from xml file
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


images = os.listdir(os.path.join(root_path,image_folder))
images.sort()

#count tables proposed and correct tables
num_tables_prop = 0
correct_tables = 0

#amnt of pixels to expand proposed bounding box by (subtract delta from min x/y, add delta to max x/y)
delta = 5

#dictionary to track tables propsed for each image key = [b,f]NUM, ex b37 for bound_37
tables_on_page = {}

#to read csv
with open(os.path.join(root_path,"Test_Image_Detection_Results/1700_result/Detection_Results.csv"),'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0] == "image":
            continue
        if row[0][6] == 'b':
            num = 'b'+row[0][16:-4]
            im_name = "train_bound_img_"+num[1:]+"_catface.jpg"
        else:
            num = 'f'+row[0][15:-4]
            im_name = "train_full_img_"+num[1:]+"_catface.jpg"
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
                    top_left_over = (prop_overlap[0],prop_overlap[1])
                    bot_right_over = (prop_overlap[2],prop_overlap[3])
                    print("new table is more confident")
                    ''' #to see proposed table
                    image = cv2.imread(os.path.join(root_path,image_folder,im_name))
                    cv2.rectangle(image,top_left,bot_right,(0,255,0), 3)
                    cv2.rectangle(image,top_left_over,bot_right_over,(0,0,255), 3)
                    image = cv2.resize(image,(600,800))
                    cv2.imshow("image",image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''
                else:
                    print("overlap and less confident")
                    '''#to see proposed table
                    image = cv2.imread(os.path.join(root_path,image_folder,im_name))
                    cv2.rectangle(image,top_left,bot_right,(0,0,255), 3)
                    cv2.rectangle(image,top_left_over,bot_right_over,(0,255,0), 3)
                    image = cv2.resize(image,(600,800))
                    cv2.imshow("image",image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''

            else: #add new key
                tables_on_page[num] = [[proposed,confidence]]


for num in tables_on_page:
    print("Checking ", num)
    for prop in tables_on_page[num]:
        ##get corresponding xml to calculate IoU score
        if num[0]=='b':
            xml_name = "train_bound_xml_"+num[1:]+".xml"
            im_name = "train_bound_img_"+num[1:]+"_catface.jpg"
            xml_folder = "xml_bound_train"
        else:
            xml_name = "train_full_xml_"+num[1:]+".xml"
            im_name = "train_full_img_"+num[1:]+"_catface.jpg"
            xml_folder = "xml_full_train"

        maxIOU = 0 #find max of all tables in xml compared to proposed

        #read in bounds
        with open(os.path.join(root_xml,xml_folder,xml_name),'r') as xml:
            tree = ET.parse(xml)

            root = tree.getroot()
            for table in root:

                points = table[0].attrib["points"]
                xml_locs = get_coordinates(points)

                iou = calc_IoU(xml_locs,prop[0])
                if iou > maxIOU:
                    maxIOU = iou
         #if want to see proposed tables
        top_left = (prop[0][0],prop[0][1])
        bot_right = (prop[0][2],prop[0][3])
        '''
        image = cv2.imread(os.path.join(root_path,image_folder,im_name))
        cv2.rectangle(image, top_left, bot_right, (255,0,0), 3)
        image = cv2.resize(image,(600,800))
        cv2.imshow("image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''


        #if maxIOU > threshold, correctly identifies table
        if maxIOU > 0.7:
            ''' #if want to see the tables propsed with IoU in (0.7,0.8]
            if maxIOU <= 0.8:
                image = cv2.imread(os.path.join(root_path,image_folder,im_name))
                top_left = (prop[0][0],prop[0][1])
                bot_right = (prop[0][2],prop[0][3])
                cv2.rectangle(image, top_left, bot_right, (255,0,0), 3)
                image = cv2.resize(image,(600,800))
                cv2.imshow("image",image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            '''
            correct_tables+=1
        num_tables_prop+=1


#count ground truth tables
num_tables_xml=0
for img in images[1:]:
    if img[6] == 'b': #if bound
        num = img[16:-12]
        xml = "train_bound_xml_"+num+".xml"
        xml_folder = "xml_bound_train"
    else: #if full
        num = img[15:-12]
        xml = "train_full_xml_"+num+".xml"
        xml_folder = "xml_full_train"



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
