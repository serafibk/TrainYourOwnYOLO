import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import csv

root = r"/Users/serafinakamp/Desktop/TableExt/datasheet-scrubber/src/Table_Extraction_Weight_Creation"

xml_folder = os.path.join(root, "xml_bound_train")

xmls = os.listdir(xml_folder)
xmls.sort()

num_train=0
with open("Annotations-export.csv","w") as file:
    writer = csv.writer(file) #open csv file
    writer.writerow(["image","xmin","ymin","xmax","ymax","label"])
    for xml_file in xmls[:1200]:
        num = xml_file[16:-4]
        if num[0]=='2':
            print("SKIPPED")
            continue
        num_train+=1
        im_name = "train_bound_img_"+num+".jpg"

        with open(os.path.join(xml_folder,xml_file),'r') as xml:
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


                writer.writerow([im_name,min_vals[0],min_vals[1],max_vals[0],max_vals[1],"Table"])
print(num_train)
