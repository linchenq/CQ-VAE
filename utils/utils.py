import numpy as np
import matplotlib.pyplot as plt

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
    
    
    
