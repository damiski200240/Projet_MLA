import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

class Pretraitement:
    def __init__(self, image_size=(256, 256), mean=0.5, std=0.5): # info sur la normalisation https://medium.com/@piyushkashyap045/image-normalization-in-pytorch-from-tensor-conversion-to-scaling-3951b6337bc8
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size), #resize en 256x256 comme c'est écrit dans l'article
            torchvision.transforms.ToTensor(), #doit etre transformer en tensor pour que pytorch gère bien les images
            #torchvision.transforms.Normalize(mean=self.mean, std=self.std) #normaisation entre -1 et 1
        ]) # a voir apres il faut normaliser dans le dataloader 
        # pour que lors du sauvegarde des images prétraité on aura des 
        # images naturelles (couleur)  

    def preprocess_image(self, image): #récupère une image en np.array et fait la transformation
        image = image[20:198, 0:178] #crop (d'après l'article) ca passe en 178x178
        #print("image apres crop",image.shape)
        if isinstance(image, np.ndarray):
            image = torchvision.transforms.ToPILImage()(image) #met en PIL Image si c'est un np.array parce que sinon ca marche pas avec les transform de torchvision
        return self.transform(image)

    def preprocess_batch(self, images):
        preprocessed_images = [self.preprocess_image(img) for img in images] #prétraite chaque image dans la liste
        return torch.cat(preprocessed_images, dim=0) #concatène les images prétraitées en un seul batch

    def visualize_image(self, tensor_image):
        plt.imshow(tensor_image.permute(1, 2, 0)) #le permute permet d'afficher une image qui a été tensor avant #https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
        plt.title("Image prétraitée")
        plt.axis('off')
        plt.show()