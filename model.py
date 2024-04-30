import torch
from torch import nn
from torch_geometric.nn import RGCNConv, FastRGCNConv, GCNConv, GATConv, FAConv
import torch.nn.functional as F

device = 'cpu'

class BurstGNN(nn.Module):
    def __init__(self, num_users, inters, num_num_properties, num_cat_properties,
                 embedding_dimension1=32, embedding_dimension2=16, dropout=0.3):
        super(BurstGNN, self).__init__()
        self.dropout = dropout
        self.num_users = num_users
        self.inters = inters
        self.embedding_nums = embedding_dimension1
        self.linear_relu_num = nn.Sequential(
            nn.Linear(num_num_properties, int(embedding_dimension1 / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat = nn.Sequential(
            nn.Linear(num_cat_properties, int(embedding_dimension1 / 2)),
            nn.LeakyReLU()
        )
        # map the cat and num features together
        self.linear_relu_together = nn.Sequential(
            nn.Linear(embedding_dimension1, embedding_dimension1),
            nn.LeakyReLU()
        )

        # self.burstGNN = GCNConv(num_users * inters, num_properties)
        self.FAConv = FAConv(embedding_dimension1)
        self.GCNConv = GCNConv(embedding_dimension1, embedding_dimension1)
        self.linear_relu_final1 = nn.Sequential(
            nn.Linear(embedding_dimension2, embedding_dimension2 // 2),
            nn.LeakyReLU()
        )
        self.linear_relu_label = nn.Linear(embedding_dimension2 // 2, 2)

    def forward(self, num_prop, cat_prop, tweet_range_list, edge_index, re_index):
        num = self.linear_relu_num(num_prop)
        cat = self.linear_relu_cat(cat_prop)
        x = torch.cat([num, cat], dim=1)
        # mapping together
        x = self.linear_relu_together(x)
        # user FAConv forward
        x1 = self.FAConv(x, x, edge_index)
        # x1 = self.GCNConv(x, edge_index)
        # dropout layer
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.FAConv(x1, x, edge_index)
        # x2 = self.GCNConv(x1, edge_index)
        num_users = self.num_users
        inters = self.inters
        # cat the features of the same user
        # change the aggregate function
        num0 = x2.shape[1]
        # 平方之后求和的结果会更加符合burst感知的感觉
        x2 = x2 ** 2 + 1e-8
        # loss出现NAN，可能需要加上一个很小的值然后计算
        x2 = x2 ** 0.5
        x3 = [torch.sum(x2[tweet_range_list[i]: tweet_range_list[i + 1]], dim=0).reshape(1, num0)
              for i in range(len(tweet_range_list) - 1)]
        # print(len(x3))
        '''
        x3 = []
        for i in range(len(tweet_range_list) - 1):
            print(i, tweet_range_list[i], tweet_range_list[i + 1])
            a = x2[tweet_range_list[i]: tweet_range_list[i + 1]]
            print(a.shape)
            print(tweet_range_list[i + 1] - tweet_range_list[i])
            x3.append(x2[tweet_range_list[i]: tweet_range_list[i + 1], ].reshape(1, num0 * (tweet_range_list[i + 1] - tweet_range_list[i])))
        '''
        # add user feature those don't have posted tweets
        x3 = torch.cat(x3)
        # print(x3.shape)
        x3 = torch.cat([x3, torch.zeros((self.num_users - x3.shape[0], num0),
                                        dtype=torch.float32).to(device)])
        # rearrange
        x3 = x3[re_index]

        # get label
        x = self.linear_relu_final1(x3)
        x = self.linear_relu_label(x)
        return x


class BurstBotRGCN(nn.Module):
    def __init__(self, num_users, inters, num_num_properties, num_cat_properties,
                 embedding_dimension1=32, embedding_dimension2=16, dropout1=0.3,
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                 embedding_dimension=128, dropout2=0.3, show_output=False):
        super(BurstBotRGCN, self).__init__()
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.num_users = num_users
        self.inters = inters
        self.embedding_nums = embedding_dimension1
        self.show_output = show_output
        # 每个时间片上使用的GNN模块
        # 初步版本先不包括这个部分，直接加一起得到了用户推文特征
        # self.shareGNN = GCNConv(num_users, num_properties)
        # 映射num properties以及cat properties
        self.linear_relu_num = nn.Sequential(
            nn.Linear(num_num_properties, int(embedding_dimension1 / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat = nn.Sequential(
            nn.Linear(num_cat_properties, int(embedding_dimension1 / 2)),
            nn.LeakyReLU()
        )
        # map the cat and num features together
        # 感觉加入这个可能没必要？
        self.linear_relu_together = nn.Sequential(
            nn.Linear(embedding_dimension1, embedding_dimension1),
            nn.LeakyReLU()
        )

        # 大图上使用的高通滤波器 先使用高通的FAConv吧
        # self.burstGNN = GCNConv(num_users * inters, num_properties)
        self.FAConv = FAConv(embedding_dimension1)
        self.GCNConv = GCNConv(embedding_dimension1, embedding_dimension1)
        # 合并每个节点feature的函数，也是调用一个线性层次？认为是合并cat之后使用一个线性层映射
        # burst graph 直接embedding相加不同节点的表示，map维度不变
        self.linear_relu_map = nn.Sequential(
            nn.Linear(embedding_dimension1, embedding_dimension2),
            nn.LeakyReLU()
        )
        # 得到最后的label的线性层
        self.linear_relu_label = nn.Linear(embedding_dimension2, 2)

        # 拼接的BotRGCN模型
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        # 得到最后label之前维度有点多，再加一个隐藏层吧
        self.linear_output_final0 = nn.Sequential(
            nn.Linear(embedding_dimension + embedding_dimension2, (embedding_dimension + embedding_dimension2) // 2),
            nn.LeakyReLU()
        )
        self.linear_output_final = nn.Sequential(
            nn.Linear((embedding_dimension + embedding_dimension2) // 2, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, num_prop_burst, cat_prop_burst, tweet_range_list, edge_index_burst, re_index,
                des, tweet, num_prop, cat_prop, edge_index_rgcn, edge_type):
        num = self.linear_relu_num(num_prop_burst)
        cat = self.linear_relu_cat(cat_prop_burst)
        x = torch.cat([num, cat], dim=1)
        show_output = self.show_output
        if show_output:
            print('init cat props:', x)
        # mapping together
        x = self.linear_relu_together(x)
        if show_output:
            print('first linear together:', x)
        # user FAConv forward
        x1 = self.FAConv(x, x, edge_index_burst)
        # x1 = self.GCNConv(x, edge_index_burst)
        # dropout layer
        x1 = F.dropout(x1, p=self.dropout1, training=self.training)
        x2 = self.FAConv(x1, x, edge_index_burst)
        # x2 = self.GCNConv(x1, edge_index_burst)
        if show_output:
            print('FAConv output:', x2)

        # num_users = self.num_users
        # inters = self.inters
        # cat the features of the same user
        # x3 = [x2[user * inters: user * inters + inters].reshape(1, inters * self.embedding_nums)
        #      for user in range(num_users)]
        # x3 = torch.cat(x3)
        # burst graph 优化之后需要通过用户发布推文数量确定推文属于哪一个用户
        # 这个地方需要更新算法
        # print(tweet_range_list[0], tweet_range_list[1])
        num0 = x2.shape[1]
        # 平方之后求和的结果会更加符合burst感知的感觉
        x2 = x2 ** 2 + 1e-8
        # loss出现NAN，可能需要加上一个很小的值然后计算
        x2 = x2 ** 0.5
        x3 = [torch.sum(x2[tweet_range_list[i]: tweet_range_list[i + 1]], dim=0).reshape(1, num0)
               for i in range(len(tweet_range_list) - 1)]
        # print(len(x3))
        '''
        x3 = []
        for i in range(len(tweet_range_list) - 1):
            print(i, tweet_range_list[i], tweet_range_list[i + 1])
            a = x2[tweet_range_list[i]: tweet_range_list[i + 1]]
            print(a.shape)
            print(tweet_range_list[i + 1] - tweet_range_list[i])
            x3.append(x2[tweet_range_list[i]: tweet_range_list[i + 1], ].reshape(1, num0 * (tweet_range_list[i + 1] - tweet_range_list[i])))
        '''
        # add user feature those don't have posted tweets
        x3 = torch.cat(x3)
        # print(x3.shape)
        x3 = torch.cat([x3, torch.zeros((self.num_users - x3.shape[0], num0),
                                        dtype=torch.float32).to(device)])
        # rearrange
        x3 = x3[re_index]
        if show_output:
            print('burst tensor:', x3)
        # get label
        x_burst = self.linear_relu_map(x3)
        if show_output:
            print('burst tensor map:', x_burst[0])
        # x = self.linear_relu_label(x)


        # get botRGCN feature
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        if show_output:
            print('botRGCN features:', x)

        x = self.linear_relu_input(x)
        if show_output:
            print('botRGCN features map:', x)
        x = self.rgcn(x, edge_index_rgcn, edge_type)
        x = F.dropout(x, p=self.dropout2, training=self.training)
        x = self.rgcn(x, edge_index_rgcn, edge_type)
        if show_output:
            print('botRGCN RGCN features:', x)
        x_rgcn = self.linear_relu_output1(x)
        if show_output:
            print('rgcn map:', x_rgcn)
        # concat two features to get the final label
        x = torch.cat([x_burst, x_rgcn], dim=1)
        if show_output:
            print('final feature cat:', x)
        x = self.linear_output_final0(x)
        x = self.linear_output_final(x)
        if show_output:
            print('final output:', x)

        # 加入一个sigmoid层进行二分类
        # x = torch.softmax(x, dim=1)
        return x


class BurstGCNBotRGCN(nn.Module):
    def __init__(self, num_users, inters, num_num_properties, num_cat_properties,
                 embedding_dimension1=32, embedding_dimension2=16, dropout1=0.3,
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                 embedding_dimension=128, dropout2=0.3, show_output=False):
        super(BurstGCNBotRGCN, self).__init__()
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.num_users = num_users
        self.inters = inters
        self.embedding_nums = embedding_dimension1
        self.show_output = show_output
        # 每个时间片上使用的GNN模块
        # 初步版本先不包括这个部分，直接加一起得到了用户推文特征
        # self.shareGNN = GCNConv(num_users, num_properties)
        # 映射num properties以及cat properties
        self.linear_relu_num = nn.Sequential(
            nn.Linear(num_num_properties, int(embedding_dimension1 / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat = nn.Sequential(
            nn.Linear(num_cat_properties, int(embedding_dimension1 / 2)),
            nn.LeakyReLU()
        )
        # map the cat and num features together
        # 感觉加入这个可能没必要？
        self.linear_relu_together = nn.Sequential(
            nn.Linear(embedding_dimension1, embedding_dimension1),
            nn.LeakyReLU()
        )

        # 大图上使用的高通滤波器 先使用高通的FAConv吧
        # self.burstGNN = GCNConv(num_users * inters, num_properties)
        self.FAConv = FAConv(embedding_dimension1)
        self.GCNConv = GCNConv(embedding_dimension1, embedding_dimension1)
        # 合并每个节点feature的函数，也是调用一个线性层次？认为是合并cat之后使用一个线性层映射
        # burst graph 直接embedding相加不同节点的表示，map维度不变
        self.linear_relu_map = nn.Sequential(
            nn.Linear(embedding_dimension1, embedding_dimension2),
            nn.LeakyReLU()
        )
        # 得到最后的label的线性层
        self.linear_relu_label = nn.Linear(embedding_dimension2, 2)

        # 拼接的BotRGCN模型
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        # 得到最后label之前维度有点多，再加一个隐藏层吧
        self.linear_output_final0 = nn.Sequential(
            nn.Linear(embedding_dimension + embedding_dimension2, (embedding_dimension + embedding_dimension2) // 2),
            nn.LeakyReLU()
        )
        self.linear_output_final = nn.Sequential(
            nn.Linear((embedding_dimension + embedding_dimension2) // 2, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, num_prop_burst, cat_prop_burst, tweet_range_list, edge_index_burst, re_index,
                des, tweet, num_prop, cat_prop, edge_index_rgcn, edge_type):
        num = self.linear_relu_num(num_prop_burst)
        cat = self.linear_relu_cat(cat_prop_burst)
        x = torch.cat([num, cat], dim=1)
        show_output = self.show_output
        if show_output:
            print('init cat props:', x[0])
        # mapping together
        x = self.linear_relu_together(x)
        if show_output:
            print('first linear together:', x[0])
        # user FAConv forward
        # x1 = self.FAConv(x, x, edge_index_burst)
        x1 = self.GCNConv(x, edge_index_burst)
        # dropout layer
        x1 = F.dropout(x1, p=self.dropout1, training=self.training)
        # x2 = self.FAConv(x1, x, edge_index_burst)
        x2 = self.GCNConv(x1, edge_index_burst)
        if show_output:
            print('FAConv output:', x2[0])

        # num_users = self.num_users
        # inters = self.inters
        # cat the features of the same user
        # x3 = [x2[user * inters: user * inters + inters].reshape(1, inters * self.embedding_nums)
        #      for user in range(num_users)]
        # x3 = torch.cat(x3)
        # burst graph 优化之后需要通过用户发布推文数量确定推文属于哪一个用户
        # 这个地方需要更新算法
        # print(tweet_range_list[0], tweet_range_list[1])
        num0 = x2.shape[1]
        # 平方之后求和的结果会更加符合burst感知的感觉
        x2 = x2 ** 2 + 1e-8
        # loss出现NAN，可能需要加上一个很小的值然后计算
        x2 = x2 ** 0.5
        x3 = [torch.sum(x2[tweet_range_list[i]: tweet_range_list[i + 1]], dim=0).reshape(1, num0)
               for i in range(len(tweet_range_list) - 1)]
        # print(len(x3))
        '''
        x3 = []
        for i in range(len(tweet_range_list) - 1):
            print(i, tweet_range_list[i], tweet_range_list[i + 1])
            a = x2[tweet_range_list[i]: tweet_range_list[i + 1]]
            print(a.shape)
            print(tweet_range_list[i + 1] - tweet_range_list[i])
            x3.append(x2[tweet_range_list[i]: tweet_range_list[i + 1], ].reshape(1, num0 * (tweet_range_list[i + 1] - tweet_range_list[i])))
        '''
        # add user feature those don't have posted tweets
        x3 = torch.cat(x3)
        # print(x3.shape)
        x3 = torch.cat([x3, torch.zeros((self.num_users - x3.shape[0], num0),
                                        dtype=torch.float32).to(device)])
        # rearrange
        x3 = x3[re_index]
        if show_output:
            print('burst tensor:', x3[0])
        # get label
        x_burst = self.linear_relu_map(x3)
        if show_output:
            print('burst tensor map:', x_burst[0])
        # x = self.linear_relu_label(x)


        # get botRGCN feature
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        if show_output:
            print('botRGCN features:', x[0])

        x = self.linear_relu_input(x)
        if show_output:
            print('botRGCN features map:', x[0])
        x = self.rgcn(x, edge_index_rgcn, edge_type)
        x = F.dropout(x, p=self.dropout2, training=self.training)
        x = self.rgcn(x, edge_index_rgcn, edge_type)
        if show_output:
            print('botRGCN RGCN features:', x[0])
        x_rgcn = self.linear_relu_output1(x)
        if show_output:
            print('rgcn map:', x_rgcn[0])
        # concat two features to get the final label
        x = torch.cat([x_burst, x_rgcn], dim=1)
        if show_output:
            print('final feature cat:', x[0])
        x = self.linear_output_final0(x)
        x = self.linear_output_final(x)
        if show_output:
            print('final output:', x[0])

        # 加入一个sigmoid层进行二分类
        # x = torch.sigmoid(x)
        return x


class BotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN1(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN1, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        ''''
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        '''
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x = d

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN2(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN2, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        '''
        d=self.linear_relu_des(des)
        '''
        t = self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x = t

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN3(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN3, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n = self.linear_relu_num_prop(num_prop)
        '''
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x = n

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN4(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN4, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        '''
        c = self.linear_relu_cat_prop(cat_prop)
        x = c

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN12(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN12, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x = torch.cat((d, t), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN34(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN34, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.gcn1 = GCNConv(embedding_dimension, embedding_dimension)
        self.gcn2 = GCNConv(embedding_dimension, embedding_dimension)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotGAT(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotGAT, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.gat1 = GATConv(embedding_dimension, int(embedding_dimension / 4), heads=4)
        self.gat2 = GATConv(embedding_dimension, embedding_dimension)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN_4layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN_4layers, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN_8layers(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, embedding_dimension=128,
                 dropout=0.3):
        super(BotRGCN_8layers, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x
