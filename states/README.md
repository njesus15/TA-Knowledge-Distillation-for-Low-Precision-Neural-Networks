Save states in this file

## Individual Trained Accuracy
* 32 bit: 64.29%, 63.33%
* 16 bit: 61.02%, 59.83%, 60.37%
* 8 bit: 60.89%, 60.40%, 60.90%
* 4 bit: 61.13%, 60.49%
* 2 bit: 54.39%, 52.99%
* 1 bit: 36.64%, 37.36%

Example command:
`python3 main.py --epochs 160 --student resnet20 --student-wbits 1 --student-abits 1 --dataset cifar100 --cuda 1 --trial-id '1bit_indiv'`

## Model Distilled Accuracy
* 16 bit: 64.33%
* 8 bit: 61.68%
* 4 bit: 62.24%
* 2 bit: 57.80%
* 1 bit: 41.99%

## TA Trained Accuracy
* 32 bit to 16 bit:
    * temp: 5, lambda: 0.05: 64.06%
*  16 bit to 8 bit:
    * 64.06%
* 8 bit to 8 bit
    * 64.27%
* 4 bit to 2 bit
    * 57.84%
* 2 bit to 1 bit
    * 40.03% accuracy

### TA training 32-8-2-1
* 32 bit to 8 bit
    * 64.03%
* 8 bit to 2 bit
    * 57.96
* 2 bit to 1 bit
    * 40.9
### TA training 16-4-1
* 16 bit to 4 bit
    * 61.9
* 4 bit to 1 bit
    * 40.73
### TA training 32/16/8-1
* 32 bit to 1 bit
    * 37.65
* 16 bit to 1 bit
    * 39.61
* 8 bit to 1 bit
    * 41.77



Example command:
`python3 main.py --epochs 160 --teacher resnet20 --teacher-checkpoint states/indv/resnet20_32bit_indiv_acc_64_29_best.pth.tar  --teacher-wbits 32 --teacher-abits 32\
  --student resnet20 --student-wbits 16 --student-abits 16 --dataset cifar100 --trial-id '16bit_ta' --cuda 1`
