#!pip install -Uqq fastbook
#!pip install -Uqq unpackai

#from unpackai.utils import clean_error_img
#from fastbook import *
from fastai.vision.widgets import *
import torch


#learner = load_learner("/Users/samlinsen/Dropbox/biodiversity/training/BootcampAI/01/trainedBananaModel/bananaModel.pkl")
learner = torch.load("/Users/samlinsen/Dropbox/biodiversity/training/BootcampAI/01/trainedBananaModel/bananaModel.pkl",  map_location=torch.device('cpu'))
learner.eval()

uploader    = widgets.FileUpload()
output      = widgets.Output()
classify    = widgets.Button(description='Classify')
prediction  = widgets.Label()


def on_click_classify(change):
    img = PILImage.create(uploader.data[-1])
    output.clear_output()
    with output:
      display(img.to_thumb(254,254))
    pred, pred_idx, probs = learner.predict(img)
    prediction.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


classify.on_click(on_click_classify)

VBox([widgets.Label('Select your image!'),
      uploader, classify, output, prediction])
