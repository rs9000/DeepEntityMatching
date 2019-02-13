import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def loadGloveModel(gloveFile, size_emb):
    print("Loading Glove Model...")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        model[word] = embedding
    print("...Done! ", len(model), " words loaded!")
    if '<unk>' not in model and size_emb == 50:
        model['<unk>'] = np.asarray([0.072617, -0.51393,   0.4728,   -0.52202,  -0.35534,   0.34629,   0.23211, 0.23096,   0.26694,   0.41028,   0.28031,   0.14107,  -0.30212,  -0.21095, -0.10875,  -0.33659,  -0.46313,  -0.40999,   0.32764,   0.47401,  -0.43449, 0.19959,  -0.55808,  -0.34077,   0.078477,  0.62823,   0.17161,  -0.34454, -0.2066,    0.1323,   -1.8076,   -0.38851,   0.37654,  -0.50422,  -0.012446, 0.046182,  0.70028,  -0.010573, -0.83629,  -0.24698,   0.6888,   -0.17986, -0.066569, -0.48044,  -0.55946,  -0.27594,   0.056072, -0.18907,  -0.59021, 0.55559], dtype='float32')
    return model


class NLP(nn.Module):
    def __init__(self, word_embed, word_embed_size, n_attrs, device):
        super(NLP, self).__init__()

        self.n_attr = n_attrs
        self.size_embed = word_embed_size
        self.words = loadGloveModel(word_embed, word_embed_size)
        self.fc1 = nn.Linear(n_attrs, 50)
        self.fc2 = nn.Linear(50, 2)
        self.probs = nn.LogSoftmax(dim=-1)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.normal_(self.fc1.weight, std=1)
        nn.init.normal_(self.fc1.bias, std=0.01)
        nn.init.normal_(self.fc2.weight, std=1)
        nn.init.normal_(self.fc2.bias, std=0.01)

    def create_embed(self, x1):
        t1 = []

        for i in range(self.n_attr):
            count = 0
            attr = torch.zeros(self.size_embed).to(self.device)
            for token in str(x1[i]).split(" "):
                token = token.replace(".0", "")
                if token.lower() in self.words:
                    attr = attr.add(torch.tensor(self.words[token.lower()]).to(self.device))
                else:
                    attr = attr.add(torch.tensor(self.words['<unk>']).to(self.device))
                count += 1
            t1.append(attr.div(count))
        return torch.stack(t1)

    def forward(self, x1, x2):

        t1 = self.create_embed(x1)
        t2 = self.create_embed(x2)

        sim = F.cosine_similarity(t1, t2, dim=1)
        sim.requires_grad = True

        out = self.fc1(sim.unsqueeze(0))
        out = self.probs(self.fc2(out))

        return out
