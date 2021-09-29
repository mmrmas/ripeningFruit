#!pip install -Uqq fastbook
#!pip install -Uqq unpackai
#borrow from https://sparrow.dev/pytorch-quick-start-classifying-an-image/

#from unpackai.utils import clean_error_img
import PySimpleGUI as sg
import torch
import os
import io
from PIL import Image, ImageTk
import requests
from torchvision import models
import torchvision.transforms as T
from torch.autograd import Variable

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]
labels     = ['rot', 'three_days', 'two_days'] #from learner.dls.vocab


def main():
    # the model was created by torch.save(learner.model,'drive/MyDrive/saved_model_bananas_torch/bananaModel2.pt') in the notebook
    model = torch.load("/Users/samlinsen/Dropbox/biodiversity/training/BootcampAI/01/trainedBananaModelTorch/bananaModel2.pt",  map_location=torch.device('cpu'))

    #print(model)
    model.eval()
    print(model)

    normalize = T.Normalize(
       mean=[0.485, 0.456, 0.406],  #is this standard? I see the values multiple times, but should be consistent with training data
       std=[0.229, 0.224, 0.225]
    )
    preprocess = T.Compose([
       T.Resize(224),
       T.CenterCrop(224),
       T.ToTensor(),
       normalize
    ])


    layout = [
        [sg.Image(key="-IMAGE-")],
        [sg.Text(size=(40, 1), key="-TOUT-")],

        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse("Get",file_types=file_types),
            sg.Button("Clear"),
            sg.Button("Show"),
            sg.Button("Predict"),
    #        sg.Text(prediction),
        ],
    ]

    window = sg.Window("Image Viewer", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Clear":
            #window["-IMAGE-"].Update('')
            window.FindElement("-IMAGE-").Update('')
        if event == "Show":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((254,254))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
        if event == "Predict":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                #image.thumbnail((254,254))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                #window["-IMAGE-"].update(data=bio.getvalue())
                img_tensor = preprocess(image)
                img_tensor.unsqueeze_(0)
                img_variable = Variable(img_tensor)
                fc_out = model(img_variable)
                print (labels[fc_out.data.numpy().argmax()])
                window["-TOUT-"].update('life expectancy: ' + labels[fc_out.data.numpy().argmax()])

    window.close()



if __name__ == "__main__":
    main()
