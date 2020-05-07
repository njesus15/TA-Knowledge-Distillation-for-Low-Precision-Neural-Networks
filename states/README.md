Save states in this file

## Individual Trained Accuracy

### Cifar 100
* 32 bit: 63.81
* 16 bit: 60.41
* 8 bit: 60.73
* 4 bit: 60.81
* 2 bit: 53.69
* 1 bit: 37.00

### Cifar 10
* 32 bit: 89.83
* 16 bit: 87.34
* 4 bit: 87.6
* 2 bit: 85.6
* 1 bit: 73.88

Example command:
`python3 main.py --epochs 160 --student resnet20 --student-wbits 1 --student-abits 1 --dataset cifar100 --cuda 1 --trial-id '1bit_indiv'`

## Model Distilled (No TA) Accuracy

### Cifar 100
* 32 bit to 16 bit: 64.33
* 16 bit to 8 bit: 61.68
* 8 bit to 4 bit: 62.24
* 4 bit to 2 bit: 57.80
* 2 bit to 1 bit: 41.99

### Cifar 10
* 32 bit to 16 bit: 89.73
* 16 bit to 8 bit: 89.41
* 8 bit to 4 bit: 88.55
* 4 bit to 2 bit: 86.43
* 2 bit to 1 bit: 77.26

## TA Trained Accuracy

### Cifar 100
* 32 bit to 16 bit: 64.06
* 16 bit to 8 bit: 64.06
* 8 bit to 8 bit: 64.27
* 4 bit to 2 bit: 57.84
* 2 bit to 1 bit: 40.03

### Cifar 10
* 32 bit to 16 bit: 89.73
* 16 bit to 8 bit: 90.14
* 8 bit to 4 bit: 90.3
* 4 bit to 2 bit: 87.83
* 2 bit to 1 bit: 77.31

Example command:
`python3 main.py --epochs 160 --teacher resnet20 --teacher-checkpoint states/indv/resnet20_32bit_indiv_acc_64_29_best.pth.tar  --teacher-wbits 32 --teacher-abits 32\
  --student resnet20 --student-wbits 16 --student-abits 16 --dataset cifar100 --trial-id '16bit_ta' --cuda 1`

## Reformatted in Table

## Cifar 100
|Final Precision|Individually Trained|Model Distillation (No TA)|TA|
|:-:|:-:|:-:|:-:|
|32|63.81|-|-|-|
|16|60.41|64.33|64.06|
|8|60.73|61.68|64.06|
|4|60.81|62.24|64.27|
|2|53.69|57.80|57.84|
|1|37.00|41.99|40.03|

## Cifar 10

# Others

### Cifar 100
## TA training 32-8-2-1
* 32 bit to 8 bit
    * 64.03%
* 8 bit to 2 bit
    * 57.96
* 2 bit to 1 bit
    * 40.9

## TA training 16-4-1
* 16 bit to 4 bit
    * 61.9
* 4 bit to 1 bit
    * 40.73

## TA training 32/16/8-1
* 32 bit to 1 bit
    * 37.65
* 16 bit to 1 bit
    * 39.61
* 8 bit to 1 bit
    * 41.77

### Cifar 10
* 32-16-8 to 1 bit: 75.41
* 32 bit to 1 bit: 75.54
