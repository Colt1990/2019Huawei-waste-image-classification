# 2019Huawei-waste-image-classification

使用mixup后线下比分偏高，线上偏低，通过TTA可以弥补，推测是测试集有很多图片中的垃圾并没有位于图片中间位置。

model                      | epochs | Tricks                    | Scores
----                       | :---:  | :---:                     | :---: 
Resnet18  finetune         | 50     |                           | 78~79  
Densenet101 finetune       | 50     |                           | 85 
Resnext101 32*16d  finetune| 50     |                           | 91~92
Resnext101 32*16d fulltune | 50+50  |                           | 93.5~94
Resnext101 32*16d fulltune | 50+50  | Label Smoothing           | 94.1
Resnext101 32*16d fulltune | 50+50  | Label Smoothing+水平flip  | 94.5
Resnext101 32*16d fulltune | 50+100 | Label Smoothing+水平flip+CJ  | 94.68
Resnext101 32*16d fulltune | 50+100 | Label Smoothing+水平flip+CJ+TTA  |95.4
Resnext101 32*16d fulltune | 50+100 | Mixup+水平flip   |94.4
Resnext101 32*16d fulltune | 50+100 | Mixup+水平flip   |94.4
Resnext101 32*16d fulltune | 50+100 | Mixup+Label Smoothing+水平flip   |94.77


TTA:Tencrop average ,   CJ:Color jitting(0.3,0.3,0.3)
