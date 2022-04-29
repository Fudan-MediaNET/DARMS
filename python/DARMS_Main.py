import json
import scipy.signal as signal
import scipy.fftpack
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import scipy.signal as signal
import datetime
import os
import darms_model
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


class ToilFallDataset(Dataset):
    def __init__(self, motion_num):
        self.input = []
        self.fs = 500
        self.motion_num = motion_num
        self.win_len = 256

    def append(self, csi, label):
        label_one_hot = np.zeros([self.motion_num], dtype=np.float32)
        label_one_hot[int(label)] = 1

        csi_fft = np.zeros([csi_num, segment_length], dtype=np.float32)
        pca = PCA(n_components=1)
        for i in range(6):
            tmp = csi[i*30:i*30+30, :]
            csi_maincom = pca.fit_transform(np.transpose(tmp))
            f, t, Zxx = signal.stft(csi_maincom, fs=self.fs, window=('kaiser', 3), nperseg=self.win_len, noverlap=self.win_len-1, nfft=self.win_len,
                                    detrend=False, return_onesided=False, boundary='zeros', padded=True, axis=0)
            csi_fft[i*30:i*30+30, :]  = np.float32(np.abs(Zxx[0:30,0,:-1]))


        csi_fft = np.reshape(csi_fft, (1, csi_num, -1))
        csi = np.reshape(csi, (1, csi_num, -1))
        data = (torch.from_numpy(csi), torch.from_numpy(csi_fft), torch.from_numpy(label_one_hot))
        self.input.append(data)


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx]



def train(model, train_loader, epoch, epoch_num, train_size):
    model.train()
    train_loss = 0.0
    correct_cnt = 0
    all_labels = np.zeros([train_size], dtype=int)
    all_predicts = np.zeros([train_size], dtype=int)
    cnt = 0
    for i, data in enumerate(train_loader):
        (csi, csi_fft, labels_onehot) = data
        # step1
        csi = csi.cuda()
        csi_fft = csi_fft.cuda()
        labels_onehot = labels_onehot.cuda()

        predicts_onehot = model(csi, csi_fft)

        # step2
        optimizer.zero_grad()
        loss = criterion(predicts_onehot, labels_onehot)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

        predicts = torch.argmax(predicts_onehot, dim=-1)
        labels = torch.argmax(labels_onehot, dim=-1)
        correct_cnt = correct_cnt + torch.sum(predicts == labels)

        all_labels[cnt:cnt + len(labels)] = labels.cpu().detach().numpy()
        all_predicts[cnt:cnt + len(labels)] = predicts.cpu().detach().numpy()
        cnt += len(labels)

        if i % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoch_num, loss.item()))

    print('Train set: Average loss: {:.4f}'.format(train_loss / (i + 1)))
    print('Train set: Mean Accuracy: {:.4f}'.format(correct_cnt / train_size))

    return all_labels, all_predicts, train_loss/train_size


def test(model, test_loader, test_size):
    model.eval()
    test_loss = 0.0
    correct_cnt = 0
    all_labels = np.zeros([test_size], dtype=int)
    all_predicts = np.zeros([test_size], dtype=int)
    cnt = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            (csi, csi_fft, labels_onehot) = data
            csi = csi.cuda()
            csi_fft = csi_fft.cuda()
            labels_onehot = labels_onehot.cuda()
            predicts_onehot = model(csi, csi_fft)
            test_loss = test_loss + criterion(predicts_onehot, labels_onehot).item()

            predicts = torch.argmax(predicts_onehot, dim=-1)
            labels = torch.argmax(labels_onehot, dim=-1)
            correct_cnt = correct_cnt + torch.sum(predicts == labels)

            all_labels[cnt:cnt+len(labels)] = labels.cpu().detach().numpy()
            all_predicts[cnt:cnt + len(labels)] = predicts.cpu().detach().numpy()
            cnt += len(labels)
        print('Test set: Average loss: {:.4f}'.format(test_loss/(i+1)))
        print('Test set: Mean Accuracy: {:.4f}'.format(correct_cnt / test_size))
    return all_labels, all_predicts, test_loss/test_size


def cal_acc(predicts_onehot, labels_onehot, correct_cnt):
    predicts = torch.argmax(predicts_onehot, dim=-1)
    labels = torch.argmax(labels_onehot, dim=-1)
    res = correct_cnt + torch.sum(predicts == labels)
    return res


def load_data(set_name, dataset):
    print("Loading the Dataset")
    for key in m.keys():
        path = set_name + key + '/'
        files = os.listdir(path)
        for cnt, it in enumerate(files):
            all_csi = np.load(path + it)
            csi = np.zeros([30*2*ant_num_per_chain, segment_length], dtype=float)
            for i in range(ant_num_per_chain):
                csi[i : (i+1)*30, :] = all_csi[i:(i+1)*30, :]
                csi[i+30*ant_num_per_chain : (i+1)*30+30*ant_num_per_chain, :] = all_csi[i+90 : (i+1)*30+90, :]
            label = m[key]
            csi = np.float32(csi)
            label = np.float32(label)
            dataset.append(csi, label)


def main(model):
    train_size = int(len(dataset)*0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, pin_memory=True, num_workers=nw)

    model = model.cuda()

    for epoch in range(0, epoch_num):
        train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, pin_memory=True, num_workers=nw)

        all_labels, all_predicts, train_loss = train(model, train_loader, epoch, epoch_num, train_size)
        cm = confusion_matrix(all_labels, all_predicts)
        acc = np.sum(np.diag(cm))/np.sum(cm)


        if (epoch+1)%5 == 0: #Test the model performance every 5 epoch
            all_labels, all_predicts, test_loss = test(model, test_loader, test_size)
            cm = confusion_matrix(all_labels, all_predicts)






if __name__ == '__main__':
    m = {'empty': 0, 'walk': 1, 'sit': 2, 'stand': 3, 'stooll': 4, 'stoolr': 5, 'stoolf': 6}

    ant_num_per_chain = 3
    csi_num = ant_num_per_chain*30*2

    s = set()
    for value in m.values():
        s.add(value)
    dataset = ToilFallDataset(motion_num = len(s))
    segment_length = 800
    dataset_path = '../dataset/'
    load_data(dataset_path, dataset)

    model = darms_model.dualchannel_net(csi_num = 180, frame_len = segment_length, motion_num = len(s))

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    print(model)

    BatchSize = 8
    epoch_num = 100

    torch.backends.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    nw = min([os.cpu_count(), BatchSize if BatchSize > 1 else 0, 8])  # number of workers

    recorder_val = {'epoch_num': [], 'loss':[], 'acc':[], 'cm':[]}
    recorder_train = {'epoch_num': [], 'loss': [], 'acc': [], 'cm': []}

    main(model)

