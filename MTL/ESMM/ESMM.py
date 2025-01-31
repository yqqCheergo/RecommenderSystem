import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from category_encoders import OrdinalEncoder
import tqdm

def load_data():
    df = pd.read_pickle('sample.pkl')
    return df


def get_embedding_size(df, embedding_dim):
    df_feature = df.drop(columns=['click', 'conversion'])

    # Get embedding layer size
    max_idxs = list(df_feature.max())   # 获取3个feature的最大类别数 [5, 7, 6]
    embedding_sizes = []
    for i in max_idxs:
        embedding_sizes.append((int(i + 1), embedding_dim))   # [(6, 5), (8, 5), (7, 5)]

    return embedding_sizes


class ESMMDataset(Dataset):
    def __init__(self, df):
        # Drop supervised columns
        df_feature = df.drop(columns=['click', 'conversion'])

        self.X = torch.from_numpy(df_feature.values).long()
        self.click = torch.from_numpy(df['click'].values).float()  # click label
        self.conversion = torch.from_numpy(df['conversion'].values).float()  # conversion label

        self.data_num = len(self.X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_X = self.X[idx]
        out_click = self.click[idx]
        out_conversion = self.conversion[idx]
        return out_X, out_click, out_conversion


class FeatureExtractor(nn.Module):    # Embedding Layer for encoding categorical variables
    def __init__(self, embedding_sizes):
        super(FeatureExtractor, self).__init__()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(unique_size, embedding_dim) for unique_size, embedding_dim in embedding_sizes])

    def forward(self, category_inputs):    # category_inputs.shape: [7, 3]  7条数据, 3个feature
        # category_inputs[:, i] 分别取出3列特征
        h = [embedding_layer(category_inputs[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
        # print(len(h))   # 3
        # print(h[0].shape)   # torch.Size([7, 5])
        h = torch.cat(h, dim=1)   # size = (minibatch, embedding_dim * Number of categorical variables) = (7, 5*3)
        return h   # torch.Size([7, 15])


class CTRNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CTRNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class CVRNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CVRNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class ESMM(nn.Module):
    def __init__(self, embedding_sizes):
        super(ESMM, self).__init__()
        self.feature_extractor = FeatureExtractor(embedding_sizes)

        input_dim = 0
        for _, embedding_dim in embedding_sizes:
            input_dim += embedding_dim   # 15
        self.ctr_network = CTRNetwork(input_dim)
        self.cvr_network = CVRNetwork(input_dim)

    def forward(self, inputs):
        # embedding
        h = self.feature_extractor(inputs)
        # Predict pCTR
        p_ctr = self.ctr_network(h)
        # Predict pCVR
        p_cvr = self.cvr_network(h)
        # Predict pCTCVR
        p_ctcvr = torch.mul(p_ctr, p_cvr)
        return p_ctr, p_ctcvr


def train(df):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build dataset
    dataset = ESMMDataset(df)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)    # len(train_loader) = 1

    # Build model
    embedding_sizes = get_embedding_size(df, 5)   # [(6, 5), (8, 5), (7, 5)]
    model = ESMM(embedding_sizes)
    model = model.to(device)

    # Settings
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 30

    # Start fitting
    model.train()
    for epoch in range(epochs):
        running_total_loss = 0.0
        running_ctr_loss = 0.0
        running_ctcvr_loss = 0.0
        for i, (inputs, click, conversion) in enumerate(train_loader):
        # for i, (inputs, click, conversion) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = inputs.to(device)   # torch.Size([7, 3])
            click = torch.unsqueeze(click.to(device), 1)   # [7] -> [7, 1]
            conversion = torch.unsqueeze(conversion.to(device), 1)

            # 反向传播和优化
            optimizer.zero_grad()   # 清零梯度
            # 计算损失
            p_ctr, p_ctcvr = model(inputs)
            ctr_loss = loss_fn(p_ctr, click)
            ctcvr_loss = loss_fn(p_ctcvr, conversion)
            total_loss = ctr_loss + ctcvr_loss
            total_loss.backward()   # 反向传播计算梯度
            optimizer.step()        # 更新参数

            running_total_loss += total_loss.item()
            running_ctr_loss += ctr_loss.item()
            running_ctcvr_loss += ctcvr_loss.item()

        running_total_loss = running_total_loss / len(train_loader)
        running_ctr_loss = running_ctr_loss / len(train_loader)
        running_ctcvr_loss = running_ctcvr_loss / len(train_loader)
        print(f'epoch: {epoch + 1}, train_loss: {running_total_loss}')


def main():
    # Load data
    df = load_data()

    # Encode categorical columns
    category_columns = ['feature1', 'feature2', 'feature3']
    encoder = OrdinalEncoder(cols=category_columns, handle_unknown='impute').fit(df)
    df = encoder.transform(df)

    # Start train
    train(df)


if __name__ == '__main__':
    main()