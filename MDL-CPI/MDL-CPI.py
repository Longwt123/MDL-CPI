import pickle
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve

import csv
import os
import sys

"""BERT"""


def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = 2048
        n_layers = 3
        n_head = 8
        d_model = dim
        d_ff = 64
        d_k = 32
        d_v = 32
        vocab_size = n_word

        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.classifier = nn.Linear(2, 2)

    def forward(self, input_ids):
        # input_ids[batch_size, seq_len] like[8,1975]
        output = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)

        return output


"""GNN"""


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])

    def forward(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)


"""AE2"""


class AENet(nn.Module):
    def __init__(self, inputDim, hiddenDim, prelr, totlr):
        super(AENet, self).__init__()

        self.enfc = nn.Linear(inputDim, hiddenDim)
        self.defc = nn.Linear(hiddenDim, inputDim)

    def encoder(self, x):
        return torch.sigmoid(self.enfc(x))

    def decoder(self, zHalf):
        return torch.sigmoid(self.defc(zHalf))

    def totolTrainOnce(self, trainDataList, g, lamda):
        g = torch.autograd.Variable(g, requires_grad=False)
        trainLoader = DataLoader(
            dataset=TensorDataset(trainDataList, g),
            batch_size=1,
            shuffle=True
        )
        for x, g in trainLoader:
            x = x.float()
            zHalf = self.encoder(x)
            z = self.decoder(zHalf)
        return z


class DGNet(nn.Module):
    def __init__(self, targetDim, hiddenDim, lr=0.001):
        super(DGNet, self).__init__()

        self.dgfc = nn.Linear(targetDim, hiddenDim)

    def degradation(self, h):
        return torch.sigmoid(self.dgfc(h))

    def totalTrainDgOnce(self, hList, zHalfList, lamda):
        hList = torch.autograd.Variable(hList, requires_grad=False)
        zHalfList = torch.autograd.Variable(zHalfList, requires_grad=False)
        trainLoader = DataLoader(
            dataset=TensorDataset(hList, zHalfList),
            batch_size=1,
            shuffle=True
        )
        for h, zHalf in trainLoader:
            g = self.degradation(h)
        return g


class Autoencoder(nn.Module):
    def __init__(self, dimList, targetDim, hiddenDim=100, preTrainLr=0.001,
                 aeTotleTrainLr=0.001, dgTotleTrainLr=0.001, lamda=1.0, HTrainLr=0.1):
        super(Autoencoder, self).__init__()

        self.viewNum = 0
        self.nSample = 1
        self.lamda = lamda
        self.HTrainLr = HTrainLr
        self.aeNetList = [AENet(d, hiddenDim, preTrainLr, aeTotleTrainLr).cuda() for d in dimList]
        self.dgNetList = [DGNet(targetDim, hiddenDim, dgTotleTrainLr).cuda() for d in dimList]
        self.H = nn.Parameter(torch.FloatTensor(np.random.uniform(0, 1, [self.nSample, targetDim])))

        self.input = []
        self.output = []

    def forward(self, trainDataList, nSample=1):
        # totleTrain
        self.nSample = nSample
        self.viewNum = len(trainDataList)  # 1
        # 1.Update aenets
        g = [dgnet.degradation(self.H) for dgnet in self.dgNetList]
        for v in range(self.viewNum):
            self.aeNetList[v].totolTrainOnce(trainDataList[v], g[v], self.lamda)

        # 2.Update dgnets&AE2
        for v in range(self.viewNum):
            zHalfList = self.aeNetList[v].encoder(trainDataList[v].float())
            # 2.1 Update denets
            self.dgNetList[v].totalTrainDgOnce(self.H, zHalfList, self.lamda)

            # 2.2 Update AE2
            tmpZHalfList = torch.autograd.Variable(zHalfList, requires_grad=False)
            trainLoader = DataLoader(
                dataset=TensorDataset(self.H, tmpZHalfList),
                batch_size=100,
                shuffle=True
            )
            for h, zHalf in trainLoader:
                self.input = zHalf
                self.output = self.dgNetList[v].degradation(h)

        return self.H, self.input, self.output

    def getH(self):
        return self.H


"""MDL-CPI model"""


class ABG(nn.Module):
    def __init__(self):
        super(ABG, self).__init__()

        self.Bert = BERT()
        self.GNN = GNN()
        self.Autoencoder = Autoencoder(dimList, dimOut)

        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_cnn = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=2 * window + 1,
            stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)

        self.W_outChange = nn.ModuleList([nn.Linear(dimChange, dimChange)
                                          for _ in range(layer_output)])
        self.W_interactionChange = nn.Linear(dimChange, 2)

    def cnn(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, data, ifTrain=True):
        correct_interaction = data[-1]

        fingerprints, adjacency, words = data[:-1]

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.GNN(fingerprint_vectors, adjacency, layer_gnn)

        """update AE2"""
        fusion_vector, aeInput, aeOutput = self.Autoencoder([compound_vector])
        HTrainOptimizor = optim.Adam([self.Autoencoder.H], lr=lr_auto)
        loss = F.mse_loss(aeOutput, aeInput)
        HTrainOptimizor.zero_grad()
        loss = loss.requires_grad_()
        loss.backward()
        HTrainOptimizor.step()

        """Protein vector with BERT-CNN."""
        protein_vectors = self.Bert(words.unsqueeze(0))
        protein_vector = self.cnn(compound_vector, protein_vectors, layer_cnn)

        """update AE2"""
        fusion_vector, aeInput, aeOutput = self.Autoencoder([protein_vector])
        HTrainOptimizor = optim.Adam([self.Autoencoder.H], lr=lr_auto)
        loss = F.mse_loss(aeOutput, aeInput)
        HTrainOptimizor.zero_grad()
        loss = loss.requires_grad_()
        loss.backward()
        HTrainOptimizor.step()

        """Fusion vector with AE2."""
        # updated data
        with torch.no_grad():
            compound_vector = self.GNN(fingerprint_vectors, adjacency, layer_gnn)
            protein_vector = self.Bert(words.unsqueeze(0))
            fusion_vector = self.Autoencoder.getH()

        """Concatenate """
        cat_vector = torch.cat((compound_vector,
                                protein_vector, fusion_vector), 1)
        cat_vector = cat_vector.to(torch.float32)

        '''Predict Module'''
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_outChange[j](cat_vector))
        interaction = self.W_interactionChange(cat_vector)

        if ifTrain:
            loss = F.cross_entropy(interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss = loss.requires_grad_()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            correct = 0
            total = 0
            for data in dataset:
                (correct_labels, predicted_labels,
                 predicted_scores) = self.model(data, False)
                correct += (predicted_labels == correct_labels).sum()
                total += len(correct_labels)
                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)
            res = [T, Y, S]
            acc = correct / total
            AUC = roc_auc_score(T, S)
            precision = precision_score(T, Y)
            recall = recall_score(T, Y)
            tpr, fpr, _ = precision_recall_curve(T, S)
            PRC = auc(fpr, tpr)
        return AUC, PRC, precision, recall, acc, res

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_datafile(file_name, ceng):
    csv_reader = csv.reader(open(file_name, encoding='utf-8'))
    newdata = []
    for row in csv_reader:
        newhang = []
        for d in row:
            newhang.append(float(d))
        x = []
        for i in range(ceng):
            x.append(newhang)
        newdata.append(x)

    tmp = np.array(newdata)
    return torch.from_numpy(tmp)


def load_npy_datalist(dir_input):
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    return compounds, adjacencies, proteins, interactions, \
           fingerprint_dict, word_dict, n_fingerprint, n_word


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    dimList = [dim]
    global dimOut, lr_auto
    dimOut = 8
    lr_auto = 0.1
    dimChange = dim + dim + dimOut
    """ About """

    about = 'MDL-CPI'

    """CPU or GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print('|' + '\t' * 6 + 'torch.cuda.current_device:' + str(torch.cuda.current_device()))
        device = torch.device('cuda')
        print('|' + '\t' * 6 + 'The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('|' + '\t' * 6 + 'The code uses CPU!!!')
    print('|' + '-' * 2 + 'MDL-CPI Hyperparameters setting OVER')

    """Load preprocessed data."""
    global n_word
    dir_input = 'xxxxxxxxxx/dataset/' + DATASET + '/input/radius2_ngram3/'
    compounds, adjacencies, proteins, interactions, \
    fingerprint_dict, word_dict, n_fingerprint, n_word = load_npy_datalist(dir_input)
    print('|' + '-' * 2 + 'MDL-CPI data load OVER')

    """Create a dataset and split it into train/dev/test..."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    torch.manual_seed(1234)
    model = ABG().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_AUCs = './output/result/'
    file_model = './output/model/'
    if not os.path.exists(file_AUCs):
        os.makedirs(file_AUCs)
    if not os.path.exists(file_model):
        os.makedirs(file_model)

    file_AUCs = './output/result/' + about + 'AUCs--' + setting + '.txt'
    file_model = './output/model/' + about + 'model_' + about + setting
    AUCs = 'Epoch\tTime(sec)\t\tLoss_train\t\t\tAUC_dev\t\t\tACC_dev\t\t\t' \
           'AUC\t\t\tPRC\t\t\tPrecision\t\t\tRecall\t\t\tACC'

    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    print('|' + '-' * 2 + 'MDL-CPI model setting OVER')

    """Start training."""
    print('|' + '-' * 2 + 'MDL-CPI train START')
    print('|' + '\t' * 6 + AUCs)
    start = timeit.default_timer()

    # TAO
    results = [[], [], []]
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay  # 在训练中动态的调整学习率

        loss_train = trainer.train(dataset_train)
        AUC_dev, PRC_dev, precision_dev, recall_dev, acc_dev, res = tester.test(dataset_dev)
        AUC_test, PRC_test, precision_test, recall_test, acc_test, res = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, acc_dev,
                AUC_test, PRC_test, precision_test, recall_test, acc_test]

        tester.save_AUCs(AUCs, file_AUCs)
        tester.save_model(model, file_model)

        results[0].extend(res[0])
        results[1].extend(res[1])
        results[2].extend(res[2])

        print('|' + '\t' * 6 + '\t'.join(map(str, AUCs)))
    print('|' + '-' * 2 + 'MDL-CPI train END')

    print("results\n")
    print(results)

    print('|' + '-' * 2 + about + "ALL FINISH !!! ")
