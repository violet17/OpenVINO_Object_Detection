# OpenVINO_Object_Detection
This repo describes the whole process from dataset making to OpenVINO deployment

## Split Dataset
```
python train_test_split.py
```

## Data Augmentation
```
pip install six numpy scipy matplotlib scikit-image opencv-python imageio
pip install editdistance fast-ctc-decode nibabel nltk parasail py-cpuinfo pydicom rawpy pillow==8.1 threadpoolctl
pip install git+https://github.com/aleju/imgaug
python data_augmentation.py
```

## Convert XML to CSV
```
pip install pandas
python xml_to_csv.py
```

## Generate TFRecord
```
git clone https://github.com/tensorflow/models.git
pip install tf_slim scipy
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
unzip -o protoc-3.17.1-linux-x86_64.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd slim
python setup.py build

cd ../object_detection/builders/
python model_builder_test.py
# no error means installed tensorflow/models environment successfully

# revise labels in function class_text_to_int in generate_tfrecord.py 
cd models/research
python generate_tfrecord.py
```

## Reference
1. https://www.cnblogs.com/gezhuangzhuang/p/10613468.html
2. https://github.com/mickkky/Dataset-Augment 
3. https://www.cnblogs.com/White-xzx/p/9503203.html



