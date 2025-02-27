import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2



def create_data(num_points, imgName='Regions02.png', seed=1):
    """
    Arguments:
    num_points -- number of points to select for each centre
    imgName -- name of image to read
    Returns:
    x, y, img
    """

    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)

    rows, cols = img.shape

    np.random.seed(seed)  # set the seet

    x = np.zeros((num_points, 2))
    y = np.zeros((num_points),dtype=int)

    i0 = 0
    while i0 < num_points:
        pos = np.random.rand(2)
        col = int(pos[0]*cols)
        row = int(pos[1]*rows)

        if img[row, col] != 0:
            x[i0] = np.array([pos[1]*rows, pos[0]*cols])
            y[i0] = img[row, col]
            i0 += 1

    labels_unique = np.unique(y)
    for i0 in range(len(labels_unique)):
        y[y==labels_unique[i0]] = i0
    
    return x,y,img


def plot_data(points, labels, figure_size=[7, 5]):
    """
    Arguments:
    :param points: -- list of points
    :param labels: -- labels
    :param figure_size: figure size, (default [7,5])
    """
    col = ['r+','g+','b+','c+','m+','y+']
    
    labels_unique = np.unique(labels)

    fig = plt.figure(figsize=figure_size)
    ax = fig.subplots()
    for i0 in np.arange(labels_unique.size):
        plt.plot(points[labels==labels_unique[i0],1],points[labels==labels_unique[i0],0],col[i0])

    plt.show()

def plot_img(img, col_map=plt.cm.gray, figure_size=[3, 3]):
    """
    plot a single image
    :param img: input image
    :param col_map: color map, (default plt.cm.gray)
    :param figure_size: figure size, (default [3,3])
    :return:
    """
    fig = plt.figure(figsize=figure_size)
    ax = fig.subplots()
    ax.imshow(img, cmap=col_map)
    ax.set_axis_off()
    plt.show()
    
    

def read_data(data_type='FashionMNIST', storage_path='data'):
    """
    reads MNIST or FashionMNIST data to folder 'data' and gives back images and labels
    :param data_type: 'MNIST' or 'FashionMNIST' (<- default)
    :param storage_path: path to store data in (default 'data')
    :return:
        training_data: set of training images and labels (60'000) of size 28 x 28 (single channel)
        test_data: set of test images and labels (10'000) of size 28 x 28 (single channel)
        labels_map: dict of strings designating the label
    """
    # only at first execution data is downloaded, because it is saved in subfolder ./data;
    if data_type == 'MNIST':
        training_data = torchvision.datasets.MNIST(
            root=storage_path,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        test_data = torchvision.datasets.MNIST(
            root=storage_path,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        # labels for MNIST (just for compatibility reasons)
        labels_map = {
            0: "Zero",
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
            9: "Nine",
        }
    else:
        training_data = torchvision.datasets.FashionMNIST(
            root=storage_path,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        test_data = torchvision.datasets.FashionMNIST(
            root=storage_path,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        # labels for FashionMNIST
        labels_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }

    return training_data, test_data, labels_map


def plot_tiles(data, cols = 10, rows = -1, figure_size = [8,8]):
    """
    plot array of images as single image
    :param x_array: array of images (being organised as ROWS!)
    :param rows/cols: an image of rows x cols - images is created (if x_array is smaller zeros are padded)
    :param figure_size: size of full image created (default [8,8])
    :return:
    """
    if rows < 0:
        rows = cols
    #data is expected to be [#images, channels, rows, columns]
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    #call helper function from torchvision (nrow is #image per row)
    img_grid = torchvision.utils.make_grid(data[:rows*cols], nrow=cols)
    #shift channel index to end
    plot_img(torch.movedim(img_grid, [0],[2]), figure_size = figure_size)
    
    
    
def plot_error(mlp_instance, y_range = [5e-3, 1]):
    #analyse error as function of epochs
    epochs = torch.arange(mlp_instance.result_data.shape[0])
    train_error = mlp_instance.result_data[:,1]
    val_error = mlp_instance.result_data[:,3]
    
    plt.semilogy(epochs, train_error, label="train")
    plt.semilogy(epochs, val_error, label="validation")
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    xmax = epochs[-1]
    ymin = y_range[0]
    ymax = y_range[1]
    plt.axis([0,xmax,ymin,ymax])
    
    plt.legend()
    plt.show() 
    
    
def plot_cost(mlp_instance, y_range = [0.2, 1.]):
    #analyse cost as function of epochs
    epochs = torch.arange(mlp_instance.result_data.shape[0])
    train_costs = mlp_instance.result_data[:,0]    
    val_costs = mlp_instance.result_data[:,2]    

    plt.semilogy(epochs, train_costs, label="train")
    plt.semilogy(epochs, val_costs, label="validation")
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    xmax = epochs[-1]
    ymin = y_range[0]
    ymax = y_range[1]
    plt.axis([0,xmax,ymin,ymax])

    plt.legend()
    plt.show() 