import numpy as np
from numpy.core.fromnumeric import sort
import torch
import PIL 
import io, base64
import torch.nn as nn
from itertools import islice
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

class CLF(nn.Module):
    def __init__(self):
        super(CLF,self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def url_prediction(url):
    im = Image.open(io.BytesIO(base64.b64decode(url.split(',')[1])))
    img = 255- np.array(im.convert("L").resize((28,28)))
    model = CLF()
    model.load_state_dict(torch.load("model_weights.pth"))
    img_t = torch.from_numpy(img).unsqueeze(dim=0).unsqueeze(dim=0).type(torch.float32)
    model.eval()
    prediction = model(img_t)
    prediction_a = {}
    for i in range(len(prediction[0])):
        prediction_a[i] = round(prediction[0][i].item(),3)
        # prediction_a.append(round(prediction[0][i].item(),3))
    return prediction_a


app = Flask(__name__)
CORS(app)
cors = CORS(app,resources={
    r"/*":{
        "origins":"localhost"
    }
})
# @cross_origin(origin='http://localhost')
# app.config['CORS_HEADERS'] = 'Content-Type', ""


@app.route("/",methods=["POST"])
def index():
    temp = request.json.get("url")
    preds = url_prediction(temp)
    sorted_preds = dict(sorted(preds.items(), key=lambda item: item[1],reverse=True))
    print(sorted_preds)
    # response = jsonify(message="Simple server is running")
    # response.headers.add("Access-Control-Allow-Origin", "*")
    top = max(sorted_preds, key=sorted_preds.get)
    return str(top)

if __name__ =="__main__":
    app.run(debug=True)

