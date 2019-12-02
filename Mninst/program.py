import neuronesReseau as nr 
import numpy as np 
import matplotlib.pyplot as plt 


with np.load("C:\\Users\\utilisateur\\Desktop\\Projet\\Neural Lague\\mnist.npz") as df :
    t_images = df['training_images']
    t_labels = df['training_labels']
    print(" t_images shape = ",t_images.shape,"-- t_labels shape = ",t_labels.shape)

for i in range(3):
    plt.imshow(t_images[i].reshape(28,28), cmap='gray')
    plt.show()
    

layer_sizes = (784,5,10)


net = nr.NeuroneReseau(layer_sizes)
net.print_accuracy(t_images,t_labels)
