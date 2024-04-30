from model import BotRGCN, BurstGNN, BurstBotRGCN, BurstGCNBotRGCN
from Dataset import Twibot22, TwiBotBurstGraph
import torch
from torch import nn
from utils import accuracy, init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
import matplotlib.pyplot as plt
import sys


device = 'cpu'
# inters use to generate burst graph
inters = 50
# loss function used
# LOSS = 'BCE'
LOSS = 'cross_entropy'
# use direct graph or indirect graph
useBoth = False
# used edge type
useReply, useRetweet, useQuote = True, True, True
# embedding_size, dropout, lr, weight_decay = 32, 0.1, 1e-2, 5e-2
# embedding_size1, embedding_size2, dropout1, dropout2, lr, weight_decay = 32, 16, 0.1, 0.1, 1e-3, 5e-3
embedding_size1, embedding_size2, embedding_size, dropout1, dropout2, lr, weight_decay = 16, 16, 32, 0.3, 0.3, 0.01, 0.005
num_users = 8694
train_epoch = 10
test_epoch = 20
LOG = False

# add cmd params:
cmd0 = 'python '
cmd0 += ' '.join(sys.argv)

for arg in sys.argv:
    if arg[-3:] == '.py':
        continue
    elif arg[0] == 'i':
        inters = int(arg[2:])
        print('get param inters:', inters)
    elif arg[0] == 'u':
        if arg[1] == '1':
            useReply = (arg[3:] == 'True')
        elif arg[1] == '2':
            useRetweet = (arg[3:] == 'True')
        elif arg[1] == '0':
            useBoth = (arg[3:] == 'True')
        else:
            useQuote = (arg[3:] == 'True')
    elif arg[0] == 'e':
        if arg[1] == '1':
            embedding_size1 = int(arg[3:])
            print('get embedding_size1', embedding_size1)
        elif arg[1] == '2':
            embedding_size2 = int(arg[3:])
            print('get embedding_size2', embedding_size2)
        else:
            embedding_size = int(arg[2:])
            print('get embedding_size', embedding_size)
    elif arg[0] == 'd':
        if arg[1] == '1':
            dropout1 = float(arg[3:])
            print('get dropout1', dropout1)
        else:
            dropout2 = float(arg[3:])
            print('get dropout2', dropout2)
    elif arg[0] == 'l':
        lr = float(arg[3:])
        print('get learning_rate', lr)
    elif arg[0] == 'w':
        weight_decay = float(arg[2:])
        print('get weight_decay', weight_decay)
    else:
        continue

# 最后输出的时候包含args的信息

# add loss weight to deal with data imbalance
loss_weight = torch.tensor([1, 9], dtype=torch.float32).to(device)

root_burst = './processed_burst_data/'
root_rgcn = './processed_data/'
root_output = 'output/'

# dataset_burst details
dataset_burst = TwiBotBurstGraph(root=root_burst, device=device, inters=inters)
burst_num_tensor, burst_cat_tensor, tweet_range_list,\
    edgeOrigin, edgeReply, edgeRetweet, edgeQuote, \
    labels, train_idx, val_idx, test_idx, re_index = dataset_burst.dataloader()

# dataset botRGCN details
dataset_rgcn = Twibot22(root=root_rgcn, device=device, process=False, save=False)
des_tensor, tweets_tensor, num_prop, category_prop, edge_index_rgcn, edge_type, labels, train_idx, val_idx, test_idx = dataset_rgcn.dataloader()

# print(labels)
# exit(0)
burst_cat_tensor = burst_cat_tensor.to(device)
burst_num_tensor = burst_num_tensor.to(device)

num_properties = burst_num_tensor.shape[1]
cat_properties = burst_cat_tensor.shape[1]

print('get tweet num properties:', num_properties)
print('get tweet cat properties:', cat_properties)
# exit()

edge_index_burst = edgeOrigin
# 添加双向的连边
if useBoth:
    edge_index_burst = torch.cat((edgeOrigin, torch.flip(edgeOrigin, dims=[0])), dim=1)
# edge_index_burst_both = torch.cat((edgeOrigin, torch.flip(edgeOrigin, dims=[0])), dim=1)
if useReply:
    print('burst graph add reply edge.')
    edge_index_burst = torch.cat([edge_index_burst, edgeReply], dim=1)
if useRetweet:
    print('burst graph add retweet edge.')
    edge_index_burst = torch.cat([edge_index_burst, edgeRetweet], dim=1)
if useQuote:
    print('burst graph add quote edge.')
    edge_index_burst = torch.cat([edge_index_burst, edgeQuote], dim=1)


# model = BotRGCN(cat_prop_size=3, embedding_dimension=embedding_size).to(device)
# model = BurstGNN(num_users, inters, num_properties, cat_properties,
#                  embedding_dimension1=embedding_size1, embedding_dimension2=embedding_size2,
#                  dropout=dropout).to(device)
model = BurstBotRGCN(num_users, inters, num_properties, cat_properties,
                     embedding_dimension1=embedding_size1, embedding_dimension2=embedding_size2,
                     dropout1=dropout1, cat_prop_size=3, embedding_dimension=embedding_size,
                     dropout2=dropout2).to(device)
# model = BurstGCNBotRGCN(num_users, inters, num_properties, cat_properties,
#                         embedding_dimension1=embedding_size1, embedding_dimension2=embedding_size2,
#                         dropout1=dropout1, cat_prop_size=3, embedding_dimension=embedding_size,
#                         dropout2=dropout2).to(device)
'''
self, num_users, inters, num_num_properties, num_cat_properties,
                 embedding_dimension1=32, embedding_dimension2=16, dropout1=0.3,
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                 embedding_dimension=128, dropout2=0.3
'''
# add cross entropy loss weight info
if LOSS == 'BCE':
    loss = nn.BCEWithLogitsLoss(weight=loss_weight)
else:
    loss = nn.CrossEntropyLoss(weight=loss_weight)
label_2d = torch.load(root_burst + 'label_2d.pt').to(device)
# loss = nn.CrossEntropyLoss(weight=loss_weight)
# seems that amsgrad=True can get a better loss result
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=lr, weight_decay=weight_decay, amsgrad=True)

def write_log(data):
    with open(root_output + 'log_final.txt', mode='a') as f:
        f.write(str(data) + '\n')

def train(epoch):
    model.train()
    # output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    # perhaps just change the output sentence is enough
    # set the input to device
    # output = model(burst_num_tensor, burst_cat_tensor, edge_index)
    output = model(burst_num_tensor, burst_cat_tensor, tweet_range_list, edge_index_burst, re_index,
                   des_tensor, tweets_tensor, num_prop, category_prop, edge_index_rgcn, edge_type)
    # loss nan——可能要加上一个很小的值
    # print(output.shape, labels.shape)
    if LOSS == 'BCE':
        loss_train = loss(output[train_idx], label_2d[train_idx])
    else:
        loss_train = loss(output[train_idx], labels[train_idx])
    # print(labels)
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()), )
    if LOG and epoch % 10 == 9:
        s0 = 'train epoch' + str(epoch + 1) + ' loss:' + str(loss_train.item()) + \
            ' acc_train:' + str(acc_train.item()) + ' acc_val:' + str(acc_val.item())
        write_log(s0)
    return acc_train, loss_train


def test():
    model.eval()
    # output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    # perhaps just change the output sentence is enough
    # output = model(burst_num_tensor, burst_cat_tensor, edge_index)
    output = model(burst_num_tensor, burst_cat_tensor, tweet_range_list, edge_index_burst, re_index,
                   des_tensor, tweets_tensor, num_prop, category_prop, edge_index_rgcn, edge_type)
    # torch.save(output, root_burst + 'inters' + str(inters) + 'test_output.pt')
    # torch.save(labels, root_burst + 'inters' + str(inters) + 'labels_output.pt')
    if LOSS == "BCE":
        loss_test = loss(output[test_idx] + 1e-8, label_2d[test_idx])
    else:
        loss_test = loss(output[test_idx] + 1e-8, label_2d[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output = output.max(1)[1].to('cpu').detach().numpy()
    label = labels.to('cpu').detach().numpy()
    f1 = f1_score(label[test_idx], output[test_idx])
    precision = precision_score(label[test_idx], output[test_idx])
    recall = recall_score(label[test_idx], output[test_idx])
    fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
    Auc = auc(fpr, tpr)
    print("Test set results:",
          "test_loss= {:.4f}".format(loss_test.item()),
          "test_accuracy= {:.4f}".format(acc_test.item()),
          "precision= {:.4f}".format(precision.item()),
          "recall= {:.4f}".format(recall.item()),
          "f1_score= {:.4f}".format(f1.item()),
          # "mcc= {:.4f}".format(mcc.item()),
          "auc= {:.4f}".format(Auc.item()),
          )
    if LOG:
        s0 = 'test: ' + 'test_loss:' + str(loss_test.item()) + \
            ' test_acc:' + str(acc_test.item()) + ' precision:' + str(precision.item()) + \
            ' recall:' + str(recall.item()) + ' f1_score:' + str(f1.item()) + \
            ' auc:' + str(Auc.item())
        write_log(s0)
    return loss_test, acc_test, precision, recall, f1, Auc


if LOG:
    write_log(cmd0)
    write_log('results:')

model.apply(init_weights)

loss_list = []
acc_list = []
precision_list = []
recall_list = []
f1_list = []
auc_list = []
eval_list = [loss_list, acc_list, precision_list, recall_list, f1_list, auc_list]

f1_0 = 0.4

for epoch0 in range(test_epoch):
    for epoch in range(train_epoch):
        train(epoch0 * train_epoch + epoch)
    eval0 = test()
    # save the best performance model
    f1 = eval0[4]
    if f1 > f1_0:
        name = 'f1' + str(f1)[2:6] + 'e1_' + str(embedding_size1) + 'e2_' + str(embedding_size2) \
               + 'e_' + str(embedding_size) + 'd1_' + str(dropout1) + 'd2_' + str(dropout2)  \
               + 'lr' + str(lr) + 'w' + str(weight_decay) + 'u0' + str(useBoth) + '_state_dict.pt'
        torch.save(model.state_dict(), root_output + name)
        f1_0 = f1
    for i in range(6):
        eval_list[i].append(eval0[i].item())
    # plt.plot([epoch0] * 6, [loss_test, acc_test, precision, recall, f1, auc])


print('loss_list:', eval_list[0])
print('acc_list:', eval_list[1])
print('precision_list:', eval_list[2])
print('recall_list:', eval_list[3])
print('f1_list:', eval_list[4])
print('auc_list:', eval_list[5])
print('best f1:', f1_0)
print('lr', lr, 'weight_decay', weight_decay)

if LOG:
    write_log('loss_list ' + str(eval_list[0]))
    write_log('acc_list ' + str(eval_list[1]))
    write_log('precision_list ' + str(eval_list[2]))
    write_log('recall_list:' + str(eval_list[3]))
    write_log('f1_list:' + str(eval_list[4]))
    write_log('auc_list:' + str(eval_list[5]))