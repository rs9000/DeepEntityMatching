from ntm import NLP
from tensorboardX import SummaryWriter
import pandas as pd
import torch
import recordlinkage
from sklearn.utils import shuffle

def get_label(map, v1, v2):
    label = torch.LongTensor([0]).cuda()
    loc1 = map[map["idACM"] == v1]
    loc2 = map[map["idDBLP"] == v2]
    if not loc1.empty and not loc2.empty:
        if loc1.index[0] == loc2.index[0]:
            label[0] = 1

    return label


def validation(model, df1, df2, map, candidate_links):

    true_pos, true_neg = 0, 0
    false_neg, false_pos = 0, 0
    acc = 0

    for i in range(len(candidate_links)):
        ix1 = candidate_links[i][0]
        ix2 = candidate_links[i][1]

        label = get_label(map, df1.values[ix1][0], df2.values[ix2][0])
        out = model(df1.values[ix1], df2.values[ix2])

        _, label_pred = torch.max(out, 1)

        if label.item() == label_pred.item():
            acc += 1

        if label.item() == 0 and label_pred.item() == 0:
            true_neg += 1
        elif label.item() == 0 and label_pred.item() == 1:
            false_pos += 1

        if label.item() == 1 and label_pred.item() == 1:
            true_pos += 1
        elif label.item() == 1 and label_pred.item() == 0:
            false_neg += 1

    acc = acc / len(candidate_links)
    return acc, false_neg, false_pos, true_neg, true_pos


def train(model, df1, df2, map, train_set, test_set):

    print("Start training...")
    epoch = 0

    while epoch < 50:
        print("Epoch: " + str(epoch))
        total_loss = 0
        loss = 0
        for i in range(len(train_set)):
            ix1 = train_set[i][0]
            ix2 = train_set[i][1]

            label = get_label(map, df1.values[ix1][0], df2.values[ix2][0])
            out = model(df1.values[ix1], df2.values[ix2])

            weight = torch.tensor([0.5, 1]).cuda()
            criterion = torch.nn.NLLLoss(weight=weight).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            loss += criterion(out, label)
            total_loss += criterion(out, label).item()
            if (i % 20) == 0 and i != 0:
                loss = loss / 20
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0

        total_loss = total_loss / len(train_set)
        print("Tot Loss: " + str(total_loss))

        if epoch % 1 == 0:
            acc, false_neg, false_pos, true_neg, true_pos = validation(model, df1, df2, map, test_set)
            print("Accuracy: " + str(acc))
            print("#True Positive: " + str(true_pos) + " #FP: " + str(false_pos))
            print("#True Negative: " + str(true_neg) + " #FN " + str(false_neg))

        torch.save(model.state_dict(), "checkpoint.pt")
        epoch += 1


if __name__ == "__main__":
    # writer = SummaryWriter()

    df1 = pd.read_csv("C://Users//rsdic//Desktop//database//dataset//f1_parse.csv", ";")
    df2 = pd.read_csv("C://Users//rsdic//Desktop//database//dataset//f2_parse.csv", ";")
    map = pd.read_csv("C://Users//rsdic//Desktop//database//dataset//map.csv", ";")

    indexer = recordlinkage.Index()
    indexer.sortedneighbourhood('title', window=5)
    candidate_links = shuffle(indexer.index(df1, df2))
    train_set = candidate_links[:int(3 * len(candidate_links) / 4)]
    test_set = candidate_links[int(3 * len(candidate_links) / 4):]

    model = NLP().cuda()
    print(model)

    model.load_state_dict(torch.load("checkpoint.pt"))

    train(model, df1, df2, map, train_set, test_set)
