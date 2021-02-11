import os
import numpy as np
import torch
import torchvision as tv

def load_mnist():
    root_mnist = os.path.expanduser("~/data/datasets/torch/mnist")
    
    train_mnist_dataset = tv.datasets.MNIST(root_mnist, train=True, transform=None,
                                                       target_transform=None, download=True)
    test_mnist_dataset = tv.datasets.MNIST(root_mnist, train=False, transform=None, 
                                                      target_transform=None, download=True)
    
    x_train_mnist = np.array(train_mnist_dataset.data, dtype=np.float)
    x_train_mnist = x_train_mnist.reshape(x_train_mnist.shape[0],-1)/255
    
    x_test_mnist = np.array(test_mnist_dataset.data, dtype=np.float)
    x_test_mnist = x_test_mnist.reshape(x_test_mnist.shape[0],-1)/255
    
    y_train_mnist = np.array(train_mnist_dataset.targets, dtype=np.int)
    y_test_mnist  = np.array(test_mnist_dataset.targets, dtype=np.int)
    
    
    return {'x_train': x_train_mnist, 'x_test': x_test_mnist,
            'y_train': y_train_mnist, 'y_test':y_test_mnist}

if __name__ == '__main__':
    load_mnist()