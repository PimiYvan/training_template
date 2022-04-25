import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


def plot(train_loss, val_loss):
    plt.title("Training results: Acc")
    plt.plot(val_loss,label='val_acc')
    plt.plot(train_loss, label="train_acc")
    plt.legend()
    increment = random.randint(0, 50000)
    plt.savefig('./figures/train_res'+ str(increment)+ '.png')
    plt.show()

def img_display(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg