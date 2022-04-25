
import argparse
import models
import torchvision.transforms as transforms
from PIL import Image

import warnings

# Prediction
def image_transform(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor

def predict(imagepath, verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = './model/model_ok.pth'
    try:
        checks_if_model_is_loaded = type(model)
    except:
        model = models.resnet()
    model.eval()
    #summary(model, input_size=(3,244,244))
    if verbose:
        print("Model Loaded..")
    image = image_transform(imagepath)
    image1 = image[None,:,:,:]
    # ps= torch.exp(model(image1))
    ps= model(image1)
    topconf, topclass = ps.topk(1, dim=1)
    # print(ps)
    if topclass.item() == 1:
        return {'class':'dog','confidence':str(topconf.item())}
    else:
        return {'class':'cat','confidence':str(topconf.item())}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--image_path', required=True, help="upload the image")
mains_args = vars(parser.parse_args())

image_path = mains_args['image_path']

res=predict(image_path)
print(res)