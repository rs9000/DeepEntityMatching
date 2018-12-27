import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def loadGloveModel(gloveFile):
    print("Loading Glove Model...")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        model[word] = embedding
    print("...Done! ", len(model), " words loaded!")
    return model


class NLP(nn.Module):
    def __init__(self, word_embed, word_embed_size, n_attrs):
        super(NLP, self).__init__()

        self.n_attr = n_attrs
        self.size_embed = word_embed_size
        self.words = loadGloveModel(word_embed)
        self.fc1 = nn.Linear(n_attrs, 50)
        self.fc2 = nn.Linear(50, 2)
        self.probs = nn.LogSoftmax(dim=-1)
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
            attr = torch.zeros(self.size_embed).cuda()
            for token in str(x1[i]).split(" "):
                token = token.replace(".0", "")
                if token.lower() in self.words:
                    attr = attr.add(torch.tensor(self.words[token.lower()]).cuda())
                else:
                    attr = attr.add(torch.tensor(self.words['<unk>']).cuda())
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
