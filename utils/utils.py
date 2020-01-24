import numpy as np
import matplotlib.pyplot as plt

def show_images(images, ncols, plts=None):
    n_images = len(images)
    fig = plt.figure()
    
    if plts in None:
        for n, image in enumerate(images):
            fig.add_subplot(np.ceil(n_images/float(ncols), ncols), n+1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
        fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
        plt.show()
    
    else:
        assert(len(images) == len(plts))
        for n, (image, pts) in enumerate(zip(images, plts)):
            fig.add_subplot(np.ceil(n_images/float(ncols), ncols), n+1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            plt.plot(plts[0], plts[1], 'g-')
        fig.set_size_inches(np.array(fig.get_size_inches())*n_images)
        plt.show()
    
    
    
