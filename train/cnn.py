import torch
from torch import nn
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from sklearn import metrics
from scipy.signal import savgol_filter

#SNIP algrithm
def snip(y, iterations=20, decreasing=True):
    n = len(y)
    d = int(decreasing)

    xo = np.empty(n, dtype=np.float64)
    xy = np.array(y, dtype=np.float64)

    k = int(iterations)

    if d:
        for i in range(k, 0, -1):
            for j in range(i, n - i):
                a = xy[j]
                b = (xy[j - i] + xy[j + i]) / 2
                if b < a:
                    a = b
                xo[j] = a

            for j in range(i, n - i):
                xy[j] = xo[j]
    else:
        for i in range(1, k + 1):
            for j in range(i, n - i):
                a = xy[j]
                b = (xy[j - i] + xy[j + i]) / 2
                if b < a:
                    a = b
                xo[j] = a

            for j in range(i, n - i):
                xy[j] = xo[j]

    xo = xy.copy()

    return xo


# intensity calibration
def toc(y_data):
    return y_data / y_data.sum()


def interpolate(data, boundary, mask, kind):
    if kind != None:
        f = interp1d(data[0],
                     data[1],
                     kind=kind,
                     bounds_error=nn.functionalalse,
                     fill_value=0,
                     assume_sorted=nn.functionalalse)
    new_data = []
    if kind != None:
        for i in range(len(boundary)):
            new_data.append(f(boundary[i]))
    else:
        for i in range(len(boundary)):
            if i + 1 == len(boundary):
                break
            if ((data[0] < boundary[i + 1]) &
                (data[0] > boundary[i])).astype('int').sum() != 0:
                new_data.append(data[1][(data[0] < boundary[i + 1])
                                        & (data[0] > boundary[i])].max())
            else:
                new_data.append(0)
    new_data = np.array(new_data)

    new_data[new_data < 0] = 0
    if mask is not None:
        new_data = new_data[mask]
    return new_data


def SampleGenerator(data, boundary, housekeeping, mask=None, kind=None, shuffle=True):
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    ids = data['ID']
    x = []
    for id in ids:
        x.append(load_mass('../data/' + id + '.txt', boundary, mask, kind, housekeeping))
    x = np.array(x)

    label = data['Class']
    y = np.zeros_like(label)
    y[label == 'S'] = 1
    y = y.astype('float')
    return x, y


def load_mass(path, boundary, mask, kind,housekeeping):
    data = [[], []]

    with open(path, 'r') as file:
        lines = file.readlines()
        # remove comments
        lines = lines[8:]
        for line in lines:
            data[0].append(float(line.split(' ')[0]))
            data[1].append(float(line.split(' ')[1]))
    data = np.array(data)

    if housekeeping is not None:
        min_diff = 1e5
        for item in data[0]:
            diff = housekeeping - item
            if np.abs(diff) < np.abs(min_diff):
                min_diff = diff
        data[0] = data[0] + min_diff

    # variance stabilising
    data[1] = np.sqrt(data[1])
    # smoothing
    data[1] = savgol_filter(data[1], window_length=21,
                            polyorder=3)
    data[1][data[1] < 0] = 0
    # baseline removal
    data[1] = data[1] - snip(data[1].copy())

    #bin
    new_data = interpolate(data, boundary, mask, kind=kind)
    return new_data
    
class Network(nn.Module):
    def __init__(self,input_channels, input_sample_points, classes):
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=64, stride=1, padding=32)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(64)

        self.classifier = nn.Sequential(
            nn.Linear(7168, 700),
            nn.BatchNorm1d(700),
            nn.ReLU(inplace=True),
            nn.Linear(700, 70),
            nn.BatchNorm1d(70),
            nn.ReLU(inplace=True),
            nn.Linear(70, classes),
        )

    def forward(self, x):
        x = nn.functional.max_pool1d(nn.functional.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2)
        x = nn.functional.max_pool1d(nn.functional.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2)
        x = nn.functional.max_pool1d(nn.functional.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=2)
        x = nn.functional.max_pool1d(nn.functional.relu(self.bn4(self.conv4(x))), stride=2, kernel_size=2)
        f = x.view(x.size(0), -1)
        return self.classifier(f), f

def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


net = Network(input_channels=1, input_sample_points=1800, classes=2)
net.apply(weights_init_uniform)
criterion = nn.functional.cross_entropy
optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)

data = pd.read_csv('../data.csv')
boundary = np.arange(0, 1801) * 5 + 2000
data = data[(data.Date == '20221112') | (data.Date == '20221130-24h') | (data.Date == '20221220-24h')]
test_ptient = data.Patient.sample(80)
test_ptient_data = data[data.Patient.isin(test_ptient)]
train_data = data.drop(test_ptient_data.index).reset_index(drop=True)
test_data = test_ptient_data.reset_index(drop=True)

X_train, Y_train = SampleGenerator(train_data, boundary, housekeeping = 4428)
X_test, Y_test = SampleGenerator(test_data, boundary, housekeeping = 4428)

# start training
max_acc = 0
for epoch in range(200):
    out, feat = net(torch.tensor(X_train).unsqueeze(dim=1).float())
    out = nn.functional.softmax(out, dim=1)
    cls_loss = criterion(out, torch.tensor(Y_train).long())
    loss = cls_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.item()))
    if (epoch+1) % 10 == 0:
        net.eval()
        with torch.no_grad():
            out, feat = net(torch.tensor(X_train).unsqueeze(dim=1).float())
            out = nn.functional.softmax(out, dim=1)
            pred = (out[:,1] > out[:,0]).int()
            print(
                metrics.classification_report(Y_train, pred.numpy(),
                                              zero_division=0))
            print(metrics.confusion_matrix(Y_train, pred.numpy()))

            accuracy = metrics.accuracy_score(Y_train, pred.numpy())
            print("Accuracy: %.2f%%" % (accuracy * 100.0))


            out, feat = net(torch.tensor(X_test).unsqueeze(dim=1).float())
            out = nn.functional.softmax(out, dim=1)
            pred = (out[:,1] > out[:,0]).int()
            print(
                metrics.classification_report(Y_test, pred.numpy(),
                                              zero_division=0))
            print(metrics.confusion_matrix(Y_test, pred.numpy()))

            accuracy = metrics.accuracy_score(Y_test, pred.numpy())
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            if accuracy >max_acc:
                max_acc = accuracy
        net.train()
print(f'best acc: {max_acc}')
