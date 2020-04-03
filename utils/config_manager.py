def get_config():
    search_space = {
        "lambda_student": {
            "_type": "quniform",
            "_value":[0.05, 1.0] # , 0.05] # why is there 2 0.05?
        },
        "T_student": {
            "_type": "choice",
            "_value": [1, 2, 5, 10, 15, 20]
        },
        "seed": {
            "_type": "choice",
            "_value": [20, 31, 55]
        }
    }

    config = {
        'trial_id': '[trial id not used]',
        'seed': search_space['seed']['_value'][0],
        # Tempature and lambda knowledeg distill hyperparam
        'T_student': search_space['T_student']['_value'][2], # temp of 1 is just regular softmax
        'lambda_student': search_space['lambda_student']['_value'][0], 
    }
    return config


'''
authorName: Anonymous
experimentName: TAKD
trialConcurrency: 1
maxExecDuration: 48h
maxTrialNum: 1

#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 train.py --epochs 160 --teacher resnet110 --student resnet8 --cuda 1
  codeDir: .
  gpuNum: 1
'''