Save states in this file

## Individual Trained Accuracy
* 32 bit: 64.29%, 63.33%
* 16 bit: 61.02%, 59.83%, 60.37%
* 8 bit: 60.89%, 60.40%, 60.90%
* 4 bit: 61.13%, 60.49%
* 2 bit: 54.39%, 52.99%
* 1 bit: 36.64%, 37.36%

# Cifar10
* 32 bit: 89.83%
* 16 bit: 87.34%
* 4 bit: 87.6%
* 2 bit: 85.6%
* 1 bit: 73.88%

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
### Cifar10 (Individually Trained)
* 32 bit to 16 bit:
	* 89.73%
* 16 bit to 8 bit:
	*89.41%
* 8 bit to 4 bit:
	*88.55%
* 4 bit to 2 bit:
	*86.43 %
* 2 bit to 1 bit:
	*77.26%
* 32 bit to 1 bit:
	* 75.54%

### Cifar10 (TAs)
* 16bit TA to 8 bit:
	* 90.14%
* 16-8bit TA to 4 bit:
	* 90.3
* 16-8-4bit TA to 2 bit:
	* 87.83
* 16-8-4-2 to 1 bit:
 	* 77.31%
* 32-18-8 to 1 bit:
	* 75.41

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
