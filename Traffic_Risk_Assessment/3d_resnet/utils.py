import csv
import numpy as np
import torch
import random
import torch_utils

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


























        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    #print(n_correct_elems / batch_size)
    return n_correct_elems / batch_size

# Add function to get each classes accuracy
def calculate_accuracy_for_test(outputs, targets):
    batch_size = targets.size(0)
    print('batch size:',batch_size)
    _, pred = outputs.topk(1, 1, True)
    pred2 = pred.clone()
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    correct_critical, correct_high, correct_low, correct_medium = 0., 0., 0., 0.
    cri_high,cri_medium,cri_low=0.,0.,0.
    high_cri,high_medium,high_low=0.,0.,0.
    low_critical,low_high,low_medium=0.,0.,0.
    medium_critical,medium_high,medium_low=0.,0.,0.


    n_critical, n_high, n_low, n_medium = 0., 0., 0., 0.

    for c in range(targets.size(0)):
        if targets[c]==0:
            if pred2[c].eq(targets[c])==1:
                correct_critical = correct_critical+1
                n_critical = n_critical+1
            elif  pred2[c]==1:
                cri_high=cri_high+1
                n_critical = n_critical+1
            elif pred2[c] ==2:
                cri_low=cri_low+1
                n_critical = n_critical+1
            elif pred2[c]==3:
                cri_medium=cri_medium+1
                n_critical = n_critical+1
                
        if targets[c]==1:
            if pred2[c].eq(targets[c].to(dtype=torch.int64))==1:
                correct_high = correct_high+1
                n_high = n_high+1
            elif pred2[c]==0:
                high_cri=high_cri+1
                n_high = n_high+1
            elif pred2[c]==2:
                high_low=high_low+1
                n_high = n_high+1
            elif pred2[c]==3:
                high_medium=high_medium+1
                n_high = n_high+1
            
               
        if targets[c]==2:
            if pred2[c].eq(targets[c])==1:
                correct_low = correct_low+1
                n_low = n_low+1
            elif pred2[c]==0:
                low_critical=low_critical+1
                n_low = n_low+1
            elif pred2[c]==1:
                low_high=low_high+1
                n_low = n_low+1
            elif pred2[c]==3:
                low_medium=low_medium+1
                n_low = n_low+1
                

        if targets[c]==3:
            #print('\ntarget 4!')
            if pred2[c].eq(targets[c].to(dtype=torch.int64))==1:
                correct_medium = correct_medium+1
                n_medium = n_medium+1
            elif pred2[c]==0:
                medium_critical=medium_critical+1
                n_medium = n_medium+1
            elif pred2[c]==1:
                medium_high=medium_high+1
                n_medium = n_medium+1
            elif pred2[c]==2:
                medium_low=medium_low+1
                n_medium = n_medium+1
                
    '''
    print('in cal_acc:')
    print(n_sitting, n_standing, correct_sitting, correct_standing)
    '''
    return n_correct_elems / batch_size,n_critical, n_high, n_low, n_medium, correct_critical, correct_high, correct_low,correct_medium,cri_high,cri_low,cri_medium,high_cri,high_low,high_medium,low_critical,low_high,low_medium,medium_critical,medium_high,medium_low