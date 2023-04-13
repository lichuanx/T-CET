

import torch
import numpy as np 
import torch.nn as nn
import copy 
import traceback
import copy
import numpy as np
from foresight.pruners import predictive

def safe_hooklogdet(K):
    s, ld = np.linalg.slogdet(K)
    return 0 if (np.isneginf(ld) and s==0) else ld

def get_naswot_layerwise(model, input, target, device):
    model = copy.deepcopy(model)
    model.eval()
    batch_size = input.shape[0]
    #model.K = torch.zeros(batch_size, batch_size).cuda()
    model.K_dict = {}

    def counting_forward_hook(module, inp, out):
        try:
            out = out.view(out.size(0), -1)
            x = (out > 0).float()
            K = x @ x.t()
            if x.cpu().numpy().sum() == 0:
                # model.K_dict[module.name] = 0
                model.K_dict[module.alias] = 0
            else:
                K2 = (1.-x) @ (1.-x.t())
                matrix = K + K2
                # model.K_dict[module.name] = hooklogdet(matrix.cpu().numpy())
                abslogdet = safe_hooklogdet(matrix.cpu().numpy())
                model.K_dict[module.alias] = 0. if np.isneginf(abslogdet) else abslogdet #TODO: -inf
        except:
            pass


    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            module.alias = name
            module.register_forward_hook(counting_forward_hook)
    
    # input = input.cuda(device=device)
    input = input.to(device=device)
    with torch.no_grad():
        model(input)
    scores = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            scores.append(model.K_dict[name])
    #scores = copy.deepcopy(scores)
    del model
    del input
    return scores

def compute_nas_score(train_queue, input, target, model,n_classes, device, grad_metric='synflow'):
    nwots_scores = get_naswot_layerwise(model, input, target, device)
    measures = predictive.find_measures(model,
                            train_queue,
                            ('random', 1, n_classes), 
                            device,
                            measure_names=[grad_metric], aggregate=False)
    measures = measures[grad_metric]
    #return measures
    scores = []
    for i in range(len(measures)):
        s = measures[i].detach().view(-1) 
        if torch.std(s) == 0:
            s = torch.sum(s)
        else:
            s = torch.sum(s)/torch.std(s)

        if s != 0:
            s = torch.log(s)
            
        s = s * nwots_scores[i]

        scores.append(s.cpu().numpy())
    return np.sum(scores)