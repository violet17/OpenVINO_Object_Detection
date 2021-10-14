import os  
import random  
import time  
import shutil
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-i', '--path_to_input_dataset', type=str, required=False, 
                        default="/home/username/work/dataset/data/",
                        help="path of dataset for split ")
    args.add_argument('-o', '--path_to_output_dataset', type=str, required=False, 
                        default="/home/username/work/dataset/data_split/",
                        help="path to dataset for saving after split ")
    args.add_argument('-tv', '--trainval_percent', type=float, required=False, default=0.8,
                        help="trainval_percent ")
    args.add_argument('-tr', '--train_percent', type=float, required=False, default=0.8,
                        help="train_percent ")
    return parser.parse_args()
                        
def main():
    args = parse_arguments()

    filePath = args.path_to_input_dataset
    savePath = args.path_to_output_dataset
    xmlfilepath = filePath + "xmls/"
    

    if not os.path.exists(os.path.join(savePath,"train","images")):
        os.makedirs(os.path.join(savePath,"train","images"))
    if not os.path.exists(os.path.join(savePath,"train", "xmls")):
        os.makedirs(os.path.join(savePath,"train", "xmls"))

    if not os.path.exists(os.path.join(savePath,"valid","images")):
        os.makedirs(os.path.join(savePath,"valid","images"))
    if not os.path.exists(os.path.join(savePath,"valid", "xmls")):
        os.makedirs(os.path.join(savePath,"valid", "xmls"))

    if not os.path.exists(os.path.join(savePath,"test","images")):
        os.makedirs(os.path.join(savePath,"test","images"))
    if not os.path.exists(os.path.join(savePath,"test", "xmls")):
        os.makedirs(os.path.join(savePath,"test", "xmls"))

    trainval_percent = args.trainval_percent
    train_percent = args.train_percent
    total_xml = os.listdir(xmlfilepath)  
    num=len(total_xml)  
    xmllist=range(num)  
    tv=int(num*trainval_percent)  
    tr=int(tv*train_percent)  
    trainval= random.sample(xmllist,tv)  
    train=random.sample(trainval,tr)  
    print("train and val size",tv)  
    print("train size",tr) 

    start = time.time()

    test_num=0  
    val_num=0  
    train_num=0  

    for i in xmllist:  
        name=total_xml[i]
        if i in trainval:  #train and val set 
            if i in train: 
                directory="train"  
                train_num += 1  
            else:
                directory="valid"  
                val_num += 1  
        else:
            directory="test"  
            test_num += 1  
        filePath=os.path.join(xmlfilepath,name)  
        newfile=os.path.join(savePath, directory, "xmls", name) 
        shutil.copyfile(filePath, newfile)
        shutil.copyfile(filePath.replace('.xml','.jpg').replace("/xmls/","/images/"), newfile.replace('.xml','.jpg').replace("/xmls/","/images/"))
    
    end = time.time()  
    seconds=end-start  
    print("\ntrain total : "+str(train_num))  
    print("validation total : "+str(val_num))  
    print("test total : "+str(test_num))  
    total_num=train_num+val_num+test_num  
    print("total number : "+str(total_num))  
    print( "Time taken : {0} seconds".format(seconds))
    if num != total_num:
        print("warning: total_xml != total_num", num, total_num)


if __name__ == "__main__":
    main()
