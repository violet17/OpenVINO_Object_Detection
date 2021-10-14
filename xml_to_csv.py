import os  
import glob  
import pandas as pd  
import xml.etree.ElementTree as ET
from argparse import ArgumentParser 

def parse_arguments():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-i', '--path_to_input_dataset', type=str, required=False, 
                        default="/home/username/work/dataset/data_split_aug/",
                        help="path of dataset for augmentation ")
    return parser.parse_args()

def xml_to_csv(path):  
    xml_list = []  
    for xml_file in glob.glob(path + '/*.xml'):  
        tree = ET.parse(xml_file)  
        root = tree.getroot()
        
#        print(root.find('filename').text)
#        root.find('filename').text = xml_file.split("/")[-1][:-4] + ".jpg"#liumm
#        print(root.find('filename').text)
  
        for member in root.findall('object'): 
            value = (root.find('filename').text,  
                int(root.find('size')[0].text),   #width  
                int(root.find('size')[1].text),   #height  
                member[0].text,  
                int(member[4][0].text),  
                int(float(member[4][1].text)),  
                int(member[4][2].text),  
                int(member[4][3].text)  
                )  
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)  
    return xml_df      

def main():  
    args = parse_arguments()
    path = args.path_to_input_dataset
    for directory in ['train','test','valid']:  
        xml_path = path + directory + "/xmls/"
        xml_df = xml_to_csv(xml_path)  
        xml_df.to_csv(path + "{}_labels.csv".format(directory), index=None)
        print('Successfully converted xml to csv.')

if __name__ == "__main__":  
    main()
