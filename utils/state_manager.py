import torch

def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp['model_state_dict'])
    return model

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