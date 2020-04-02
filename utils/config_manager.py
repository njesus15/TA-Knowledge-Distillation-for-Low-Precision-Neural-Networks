    config = nni.get_next_parameter()
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    trial_id = os.environ.get('NNI_TRIAL_JOB_ID')

    train_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'device': 'cuda' if args.cuda else 'cpu',
        'trial_id': trial_id,
        'T_student': config.get('T_student'),
        'lambda_student': config.get('lambda_student'),
    }