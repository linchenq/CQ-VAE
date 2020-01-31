import numpy as np
import matplotlib.pyplot as plt
from terminaltables import AsciiTable

def show_images(images, ncols, plts=None):
    n_images = len(images)
    fig = plt.figure()
    
    if plts is None:
        for i, image in enumerate(images):
            fig.add_subplot(np.ceil(n_images/float(ncols)), ncols, i+1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
        # fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
        plt.show()
    
    else:
        assert(len(images) == len(plts))
        for i, (image, pts) in enumerate(zip(images, plts)):
            fig.add_subplot(np.ceil(n_images/float(ncols)), ncols, i+1)
            plt.subplots_adjust(wspace=0, hspace=-0.7)
            plt.axis('off')
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            plt.plot(plts[i][:,0], plts[i][:,1], 'g-')
        # fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
        plt.show()
    
def print_metrics(train, valid):
    metrics = [['Epoch', 'Train Loss', 'Valid Loss']]
    for epoch, loss in train.items():
        index = [ str(epoch),
                  loss,
                  valid[epoch] if epoch in valid.keys() else '---' ]
        metrics.append(index)
    return AsciiTable(metrics).table

def plot_loss(epoch, train, valid):
    tx, ty = list(train.keys()), list(train.values())
    vx, vy = list(valid.keys()), list(valid.values())
    
    if epoch > 30:
        tx, ty = zip(*[(x, y) for x, y in zip(tx, ty) if y < 1e5])
        vx, vy = zip(*[(x, y) for x, y in zip(vx, vy) if y < 1e5])
    
    plt.plot(tx, ty, 'r--', label='Train Loss')
    plt.plot(vx, vy, 'b-', label='Valid Loss')
    
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.show()
    
    
    
