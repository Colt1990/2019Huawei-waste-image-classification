# 2019Huawei-waste-image-classification

使用mixup后线下比分偏高，线上偏低，通过TTA可以弥补，推测是测试集有很多图片中的垃圾并没有位于图片中间位置。

model | epochs | Tricks | 
----  | --- | ---| ---
Resnet18 |50 | |78~79
Densenet101|50 | 85
