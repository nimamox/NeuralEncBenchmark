from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from .torch_device import device, dtype
from .surrogate_model import run_snn

def train(encoded_data, val_enc_data, nb_hidden, lr=2e-3, nb_epochs=10, return_weights=False, params, alpha, beta):
    params = [w1, w2]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []

    train_acc = []
    val_acc = []
    w_traj = []

    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in sparse_generator(encoded_data):
            output,_ = run_snn(x_local.to_dense(), encoded_data['batch_size'], nb_hidden, params, alpha, beta)
            m,_=torch.max(output,1)
            log_p_y = log_softmax_fn(m)
            loss_val = loss_fn(log_p_y, y_local)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f, \t %s"%(e+1,mean_loss, datetime.now().strftime("%H:%M:%S")))
        loss_hist.append(mean_loss)

        if return_weights:
          w_traj.append({'w1': w1.detach().clone(),
                         'w2': w2.detach().clone()})

        with torch.no_grad():
          # # Train accuracy
          # accs = []
          # for x_local, y_local in sparse_generator(encoded_data):
          #   output,_ = run_snn(x_local.to_dense(), encoded_data['batch_size'], nb_hidden, params, alpha, beta)
          #   m,_ = torch.max(output,1) # max over time
          #   _, am = torch.max(m,1)      # argmax over output units
          #   tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
          #   accs.append(tmp)
          # mean_acc = np.mean(accs)
          # train_acc.append(mean_acc)
          # print("\tTRAIN:{:.4f}".format(mean_acc))

          # Validation accuracy
          accs = []
          for x_local, y_local in sparse_generator(val_enc_data):
            output,_ = run_snn(x_local.to_dense(), val_enc_data['batch_size'], nb_hidden, params, alpha, beta)
            m,_ = torch.max(output,1) # max over time
            _, am = torch.max(m,1)      # argmax over output units
            tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)
          mean_acc = np.mean(accs)
          val_acc.append(mean_acc)
          print("\tVALID:{:.4f}".format(mean_acc))
    if return_weights:
      return loss_hist, train_acc, val_acc, w_traj
    return loss_hist, train_acc, val_acc
    
def compute_classification_accuracy(encoded_data, nb_hidden, params, alpha, beta):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    for x_local, y_local in sparse_generator(encoded_data, shuffle=False):
        output,_ = run_snn(x_local.to_dense(), encoded_data['batch_size'], nb_hidden, params, alpha, beta)
        m,_= torch.max(output,1) # max over time
        _, am = torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)

def init_model(nb_inputs, time_step, nb_hidden, tau_mem = 10e-3, tau_syn = 5e-3):
    global alpha, beta
    global w1, w2

    alpha   = float(np.exp(-time_step/tau_syn))
    beta    = float(np.exp(-time_step/tau_mem))

    weight_scale = 7*(1.0-beta) # this should give us some spikes to begin with

    w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

    w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

    print("init done")
    
    return [w1, w2], alpha, beta
    