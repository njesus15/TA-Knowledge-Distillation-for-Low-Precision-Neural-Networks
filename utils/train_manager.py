import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from resnet_cifar import create_cnn_model, is_resnet
from utils.dataset_loader import get_cifar100

class TrainManager(object):
    def __init__(self, student, teacher=None, train_loader=None, test_loader=None, train_config={}):
        self.student = student
        self.teacher = teacher
        self.have_teacher = bool(self.teacher)
        self.device = train_config['device']
        self.name = train_config['name']
        self.optimizer = optim.SGD(self.student.parameters(),
                                   lr=train_config['learning_rate'],
                                   momentum=train_config['momentum'],
                                   weight_decay=train_config['weight_decay'])
        if self.have_teacher:
            self.teacher.eval()
            self.teacher.train(mode=False)
            
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = train_config
    
    def train(self):
        lambda_ = self.config['lambda_student']
        T = self.config['T_student']
        epochs = self.config['epochs']
        trial_id = self.config['trial_id']
        
        max_val_acc = 0
        iteration = 0
        best_acc = 0
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.student.train()
            self.adjust_learning_rate(self.optimizer, epoch)
            loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                iteration += 1
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.student(data)
                # Standard Learning Loss ( Classification Loss)
                loss_SL = criterion(output, target) 
                loss = loss_SL
                
                if self.have_teacher:
                    teacher_outputs = self.teacher(data)
                    # Knowledge Distillation Loss
                    loss_KD = nn.KLDivLoss()(F.log_softmax(output / T, dim=1),
                                                      F.softmax(teacher_outputs / T, dim=1))
                    loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
                    
                loss.backward()
                self.optimizer.step()
            
            print("epoch {}/{}".format(epoch, epochs))
            val_acc = self.validate(step=epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                self.save(epoch, name='{}_{}_best.pth.tar'.format(self.name, trial_id))
        
        return best_acc
    
    def validate(self, step=0):
        self.student.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            acc = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # self.accuracy_history.append(acc)
            acc = 100 * correct / total
            
            print('{{"metric": "{}_val_accuracy", "value": {}}}'.format(self.name, acc))
            return acc
    
    def save(self, epoch, name=None):
        trial_id = self.config['trial_id']
        if name is None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, '{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
        else:
            torch.save({
                'model_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
            }, name)
    
    def adjust_learning_rate(self, optimizer, epoch):
        epochs = self.config['epochs']
        
        if epoch < int(epoch/2.0):
            lr = 0.1
        elif epoch < int(epochs*3/4.0):
            lr = 0.1 * 0.1
        else:
            lr = 0.1 * 0.01
        
        # update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
# This is for training single models(NOKD in paper) for baselines models (or training the first teacher)    
def train_teacher(args, train_config):
    teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
    if args.teacher_checkpoint:
        print("---------- Loading Teacher -------")
        teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
    else:
        print("---------- Training Teacher -------")
        train_loader, test_loader = get_cifar100()
        teacher_train_config = copy.deepcopy(train_config)
        teacher_name = '{}_{}_best.pth.tar'.format(args.teacher, train_config['trial_id'])
        teacher_train_config['name'] = args.teacher
        teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
        teacher_trainer.train()
        teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))

def train_student(args, train_config, teacher_model=None):
    # Student training
    if teacher_model == None:
        print("---------- No Teacher -------------")
        print("---------- Training Student -------")
    else:
        print("---------- Training Student -------")
    student_train_config = copy.deepcopy(train_config)
    train_loader, test_loader = get_cifar100()
    student_train_config['name'] = args.student
    student_trainer = TrainManager(student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
    best_student_acc = student_trainer.train()
    print(f'best_student_acc: {best_student_acc}')