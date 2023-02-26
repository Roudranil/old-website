---
title: "Audio Classification with CNN-LSTM networks"
date: 2023-01-01T09:00:37+05:30
author: "Roudranil"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: false
UseHugoToc: true
hideSummary: true
---

## Synopsis

In this project, I aim to classify 1 second long audio clips of the words "one", "two", "three", ..., "zero". The data for this project is taken from the [TensorFlow Speech Recognition Challenge](https:\\\\www.kaggle.com\\competitions\\tensorflow-speech-recognition-challenge\\overview). However I have slightly deviated from the competition, in terms of the target classes, where I have truncated the target classes to the ones I mentioned above.

I trained 3 models on the data:  
1. A baseline CNN model
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #  
==========================================================================================
BaseModel                                [128, 10]                 --  
├─convblock: 1-1                         [128, 16, 65, 23]         --  
│    └─Sequential: 2-1                   [128, 16, 65, 23]         --  
│    │    └─Conv2d: 3-1                  [128, 16, 130, 46]        160  
│    │    └─ReLU: 3-2                    [128, 16, 130, 46]        --  
│    │    └─MaxPool2d: 3-3               [128, 16, 65, 23]         --  
├─convblock: 1-2                         [128, 32, 33, 12]         --  
│    └─Sequential: 2-2                   [128, 32, 33, 12]         --  
│    │    └─Conv2d: 3-4                  [128, 32, 67, 25]         4,640  
│    │    └─ReLU: 3-5                    [128, 32, 67, 25]         --  
│    │    └─MaxPool2d: 3-6               [128, 32, 33, 12]         --  
├─convblock: 1-3                         [128, 64, 17, 7]          --  
│    └─Sequential: 2-3                   [128, 64, 17, 7]          --  
│    │    └─Conv2d: 3-7                  [128, 64, 35, 14]         18,496  
│    │    └─ReLU: 3-8                    [128, 64, 35, 14]         --  
│    │    └─MaxPool2d: 3-9               [128, 64, 17, 7]          --  
├─convblock: 1-4                         [128, 128, 9, 4]          --  
│    └─Sequential: 2-4                   [128, 128, 9, 4]          --  
│    │    └─Conv2d: 3-10                 [128, 128, 19, 9]         73,856  
│    │    └─ReLU: 3-11                   [128, 128, 19, 9]         --  
│    │    └─MaxPool2d: 3-12              [128, 128, 9, 4]          --  
├─Flatten: 1-5                           [128, 4608]               --  
├─Linear: 1-6                            [128, 10]                 46,090  
├─Softmax: 1-7                           [128, 10]                 --  
==========================================================================================
Total params: 143,242  
Trainable params: 143,242  
Non-trainable params: 0  
Total mult-adds (G): 3.90  
==========================================================================================
Input size (MB): 2.88  
Forward/backward pass size (MB): 207.40  
Params size (MB): 0.57  
Estimated Total Size (MB): 210.86  
==========================================================================================
```
2. A CRNN Model, with a LSTM following a CNNBLock
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CRNN                                     [128, 10]                 --
├─convbloc: 1-1                          [128, 128, 22]            --
│    └─Sequential: 2-1                   [128, 128, 22]            --
│    │    └─Conv1d: 3-1                  [128, 128, 44]            82,048
│    │    └─BatchNorm1d: 3-2             [128, 128, 44]            256
│    │    └─ReLU: 3-3                    [128, 128, 44]            --
│    │    └─MaxPool1d: 3-4               [128, 128, 22]            --
├─convbloc: 1-2                          [128, 128, 11]            --
│    └─Sequential: 2-2                   [128, 128, 11]            --
│    │    └─Conv1d: 3-5                  [128, 128, 22]            82,048
│    │    └─BatchNorm1d: 3-6             [128, 128, 22]            256
│    │    └─ReLU: 3-7                    [128, 128, 22]            --
│    │    └─MaxPool1d: 3-8               [128, 128, 11]            --
├─convbloc: 1-3                          [128, 256, 5]             --
│    └─Sequential: 2-3                   [128, 256, 5]             --
│    │    └─Conv1d: 3-9                  [128, 256, 11]            164,096
│    │    └─BatchNorm1d: 3-10            [128, 256, 11]            512
│    │    └─ReLU: 3-11                   [128, 256, 11]            --
│    │    └─MaxPool1d: 3-12              [128, 256, 5]             --
├─LSTM: 1-4                              [128, 256, 96]            39,552
├─Flatten: 1-5                           [128, 24576]              --
├─Sequential: 1-6                        [128, 64]                 --
│    └─Linear: 2-4                       [128, 64]                 1,572,928
│    └─ReLU: 2-5                         [128, 64]                 --
├─Linear: 1-7                            [128, 10]                 650
├─Softmax: 1-8                           [128, 10]                 --
==========================================================================================
Total params: 1,942,346
Trainable params: 1,942,346
Non-trainable params: 0
Total mult-adds (G): 2.42
==========================================================================================
Input size (MB): 2.88
Forward/backward pass size (MB): 48.31
Params size (MB): 7.77
Estimated Total Size (MB): 58.96
==========================================================================================
```
3. A Parallel CNN-LSTM model, where we have the inputs go through 5 CNN blocks and a LSTM block parallely and then they are concatenated
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ParallelNet                              [128, 10]                 --
├─CNNBLock: 1-1                          [128, 16, 63, 22]         --
│    └─Sequential: 2-1                   [128, 16, 63, 22]         --
│    │    └─Conv2d: 3-1                  [128, 16, 126, 44]        64
│    │    └─BatchNorm2d: 3-2             [128, 16, 126, 44]        32
│    │    └─ReLU: 3-3                    [128, 16, 126, 44]        --
│    │    └─MaxPool2d: 3-4               [128, 16, 63, 22]         --
├─CNNBLock: 1-2                          [128, 32, 30, 11]         --
│    └─Sequential: 2-2                   [128, 32, 30, 11]         --
│    │    └─Conv2d: 3-5                  [128, 32, 61, 22]         1,568
│    │    └─BatchNorm2d: 3-6             [128, 32, 61, 22]         64
│    │    └─ReLU: 3-7                    [128, 32, 61, 22]         --
│    │    └─MaxPool2d: 3-8               [128, 32, 30, 11]         --
├─CNNBLock: 1-3                          [128, 64, 14, 5]          --
│    └─Sequential: 2-3                   [128, 64, 14, 5]          --
│    │    └─Conv2d: 3-9                  [128, 64, 28, 11]         6,208
│    │    └─BatchNorm2d: 3-10            [128, 64, 28, 11]         128
│    │    └─ReLU: 3-11                   [128, 64, 28, 11]         --
│    │    └─MaxPool2d: 3-12              [128, 64, 14, 5]          --
├─CNNBLock: 1-4                          [128, 64, 3, 1]           --
│    └─Sequential: 2-4                   [128, 64, 3, 1]           --
│    │    └─Conv2d: 3-13                 [128, 64, 12, 5]          12,352
│    │    └─BatchNorm2d: 3-14            [128, 64, 12, 5]          128
│    │    └─ReLU: 3-15                   [128, 64, 12, 5]          --
│    │    └─MaxPool2d: 3-16              [128, 64, 3, 1]           --
├─Flatten: 1-5                           [128, 192]                --
├─RNNBlock: 1-6                          [128, 32, 256]            --
│    └─MaxPool2d: 2-5                    [128, 1, 32, 22]          --
│    └─LSTM: 2-6                         [128, 32, 256]            155,648
├─Flatten: 1-7                           [128, 8192]               --
├─Linear: 1-8                            [128, 10]                 83,850
├─Softmax: 1-9                           [128, 10]                 --
==========================================================================================
Total params: 260,042
Trainable params: 260,042
Non-trainable params: 0
Total mult-adds (G): 1.30
==========================================================================================
Input size (MB): 2.88
Forward/backward pass size (MB): 326.25
Params size (MB): 1.04
Estimated Total Size (MB): 330.17
==========================================================================================
```

## Model performance

Model performance on the validation and test sets for each of the model and the number of epochs they were trained for is listed in the table below.

<div style="width:70%;margin: auto;">
<table class="demTable">
<thead>
<tr>
	<th>Model</th>
	<th># epochs</th>
	<th>Validation accuracy</th>
	<th>Test accuracy</th>
</tr>
</thead>
<tbody>
<tr>
	<td>Baseline</td>
	<td>3&nbsp;</td>
	<td>&nbsp;26.4</td>
	<td>27.6&nbsp;</td>
</tr>
<tr>
	<td>&nbsp;CRNN</td>
	<td>3&nbsp;</td>
	<td>&nbsp;56.94</td>
	<td>56.62&nbsp;</td>
</tr>
<tr>
	<td>&nbsp;Parallel CNN-LSTM</td>
	<td>6&nbsp;</td>
	<td>84.92&nbsp;</td>
	<td>&nbsp;84.29</td>
</tr>
</tbody>
</table>
</div>

Classification report for Parallel CNN-LSTM:

<div style="width:50%;margin: auto;">

|              | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| zero         | 0.91      | 0.87   | 0.89     | 250     |
| one          | 0.71      | 0.9    | 0.79     | 248     |
| two          | 0.78      | 0.81   | 0.8      | 264     |
| three        | 0.81      | 0.91   | 0.86     | 267     |
| four         | 0.91      | 0.76   | 0.83     | 253     |
| five         | 0.86      | 0.73   | 0.79     | 271     |
| six          | 0.95      | 0.89   | 0.92     | 244     |
| seven        | 0.81      | 0.92   | 0.85     | 239     |
| eight        | 0.91      | 0.82   | 0.86     | 257     |
| nine         | 0.84      | 0.83   | 0.83     | 259     |
|              |           |        |          |         |
| **Accuracy**     |           |        | 0.84     | 2552    |
| **Macro avg**    | 0.85      | 0.84   | 0.84     | 2552    |
| **Weighted avg** | 0.85      | 0.84   | 0.84     | 2552    |

</div>

Final training was done for small number of epochs because of lack of gpu due to a gpu error, and also decreasing validation accuracy if trained after that point.

## Links

GitHub: [word-classification-with-pytorch](https://github.com/Roudranil/word-classification-with-pytorch)  
Report: [Link](https://github.com/Roudranil/word-classification-with-pytorch/blob/main/doc/report.pdf)