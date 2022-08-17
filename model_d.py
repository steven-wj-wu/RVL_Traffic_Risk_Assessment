import torch
from torch import nn

# Used model

from resnet_models import cnnlstm 
from resnet_models import cnnlstm_alexnet 
from resnet_models import cnnlstm_cbam
from resnet_models import resnet


from torch import optim
from torch.optim import lr_scheduler

def generate_model(opt,input_type):
    

    optimizer = None
    scheduler = None


    if input_type =='cbam':
        print("use cnnlstm_cbam")
        model = cnnlstm_cbam.CNNLSTM(num_classes=4)
    elif input_type=='ori':
        print("use cnnlstm")
        model = cnnlstm.CNNLSTM(num_classes=4)
    elif input_type=='3d_resnet':
        print("use 3d resnet")
        model = resnet.resnet50(
                num_classes=4,
                shortcut_type='A',
                sample_size=224,
                sample_duration=19)
        model = nn.DataParallel(model, device_ids=None)

    if not opt.no_cuda:
        model = model.cuda()

        if opt.use_mix:
            from apex import amp
            if opt.nesterov:
                dampening = 0
            else:
                dampening = opt.dampening

            optimizer = optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
            #tmax = opt.n_epochs//2 + 1
            tmax = opt.n_epochs + 1
            #scheduler = lr_scheduler.ReduceLROnPlateau(
            #optimizer, 'min', patience=opt.lr_patience)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = tmax)
            print("use apex")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            #assert opt.arch == pretrain['arch']

            pre_dict = pretrain['state_dict']
            model_dict = model.state_dict()
            #print(model_dict)
            for name, param in pre_dict.items():
                if name not in model_dict:
                    print('no dict!')
                    print(name)
                    continue
                model_dict[name].copy_(param)

            # New version load pretrain weight(With r2p1d model)
            model = nn.DataParallel(model, device_ids=None)

            
            print('hey')
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            if opt.use_mix:
                return model, parameters, optimizer, scheduler
            else:
                return model, parameters
    else:

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            #assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            '''
            '''
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    if opt.use_mix:
        return model, model.parameters(), optimizer, scheduler
    else:
        return model, model.parameters()
