import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_list):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class PLEMultiTaskModel(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden_list, towers_hidden):
        super(PLEMultiTaskModel, self).__init__()
        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden_list = experts_hidden_list
        self.towers_hidden = towers_hidden
        self.sigmoid = nn.Sigmoid()

        # expert
        self.experts_shared = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden_list) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden_list) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden_list) for i in range(self.num_specific_experts)])

        # gate
        self.dnn1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                  nn.Softmax(dim=1))
        self.dnn2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                  nn.Softmax(dim=1))

        # tower
        self.tower1 = Tower(self.experts_out, task1_dim, self.towers_hidden)
        self.tower2 = Tower(self.experts_out, task2_dim, self.towers_hidden)

    def forward(self, x):   # 输入x的维度为 torch.Size([bs, 10])
        experts_shared_output = [expert(x) for expert in self.experts_shared]
        # print('len(experts_shared_output): ', len(experts_shared_output))   # 1
        # print('experts_shared_output[0].shape: ', experts_shared_output[0].shape)   # torch.Size([bs, 16])
        experts_shared_output = torch.stack(experts_shared_output)
        # print('experts_shared_output.shape: ', experts_shared_output.shape)   # torch.Size([1, bs, 16])

        experts_task1_output = [expert(x) for expert in self.experts_task1]
        # print('len(experts_task1_output): ', len(experts_task1_output))  # 2
        # print('experts_task1_output[0].shape: ', experts_task1_output[0].shape)  # torch.Size([bs, 16])
        experts_task1_output = torch.stack(experts_task1_output)
        # print('experts_task1_output.shape: ', experts_task1_output.shape)  # torch.Size([2, bs, 16])

        experts_task2_output = [expert(x) for expert in self.experts_task2]
        experts_task2_output = torch.stack(experts_task2_output)

        # gate1
        selected1 = self.dnn1(x)
        # print('selected1.shape: ', selected1.shape)   # torch.Size([bs, 3])
        gate_expert_output1 = torch.cat((experts_task1_output, experts_shared_output), dim=0)
        # print('gate_expert_output1.shape: ', gate_expert_output1.shape)   # torch.Size([3, bs, 16])
        '''
        gate_expert_output1: [3, bs, 16] 对应abc
        selected1: [bs, 3] 对应ba
        output: bc 即 [bs, 16]
        '''
        gate1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)
        # print('gate1_out.shape: ', gate1_out.shape)   # torch.Size([bs, 16])
        final_output1 = self.tower1(gate1_out)
        # print('final_output1.shape: ', final_output1.shape)   # torch.Size([bs, 3])
        final_output1[:, 0] = self.sigmoid(final_output1[:, 0])
        final_output1[:, 2] = self.sigmoid(final_output1[:, 2])

        # gate2
        selected2 = self.dnn2(x)
        gate_expert_output2 = torch.cat((experts_task2_output, experts_shared_output), dim=0)
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        final_output2 = self.tower2(gate2_out)
        final_output2[:, 1] = self.sigmoid(final_output2[:, 1])

        return final_output1, final_output2


# 构造虚拟样本数据
torch.manual_seed(42)  # 设置随机种子以保证结果可重复
input_dim = 10
task1_dim = 3
task2_dim = 2
num_samples = 1000

X_train = torch.randn(num_samples, input_dim)

y_train_task1 = torch.empty(num_samples, task1_dim)
y_train_task1[:, 0] = torch.randint(0, 2, (num_samples,)).float()   # 是否停留
y_train_task1[:, 1] = torch.randn(num_samples)   # 停留时长
y_train_task1[:, 2] = torch.randint(0, 2, (num_samples,)).float()   # 是否点击

y_train_task2 = torch.empty(num_samples, task2_dim)
y_train_task2[:, 0] = torch.randn(num_samples)   # 点击后播放时长
y_train_task2[:, 1] = torch.randint(0, 2, (num_samples,)).float()   # 播放后是否点赞

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train_task1, y_train_task2)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 实例化模型对象
experts_hidden_list = [64, 32]
model = PLEMultiTaskModel(input_dim, 2, 1, 16, experts_hidden_list, 32)

# 定义损失函数和优化器
criterion_classifier = nn.BCELoss()
criterion_regression = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (X_batch, y_task1_batch, y_task2_batch) in enumerate(train_loader):
        # print('X_batch.shape: ', X_batch.shape)   # torch.Size([32, 10])
        # 前向传播: 获取预测值
        outputs_task1, outputs_task2 = model(X_batch)
        # print(outputs_task1.shape)   # torch.Size([32, 3])  最后一个bs是torch.Size([8, 3])
        # print(outputs_task2.shape)   # torch.Size([32, 2])  最后一个bs是torch.Size([8, 2])
        # print(y_task1_batch.shape)   # torch.Size([32, 3])  最后一个bs是torch.Size([8, 3])
        # print(y_task2_batch.shape)   # torch.Size([32, 2])  最后一个bs是torch.Size([8, 2])

        # 计算每个任务的损失
        loss_task1 = criterion_classifier(outputs_task1[:, 0], y_task1_batch[:, 0]) + criterion_regression(outputs_task1[:, 1], y_task1_batch[:, 1]) + criterion_classifier(outputs_task1[:, 2], y_task1_batch[:, 2])
        loss_task2 = criterion_regression(outputs_task2[:, 0], y_task2_batch[:, 0]) + criterion_classifier(outputs_task2[:, 1], y_task2_batch[:, 1])
        # print(f'loss_task1：{loss_task1}, loss_task2：{loss_task2}')
        total_loss = loss_task1 + loss_task2

        # 反向传播和优化
        optimizer.zero_grad()   # 清零梯度
        total_loss.backward()   # 反向传播计算梯度
        optimizer.step()        # 更新参数

        running_loss += total_loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 模型预测
model.eval()
with torch.no_grad():
    test_input = torch.randn(1, input_dim)   # 构造一个测试样本
    pred_task1, pred_task2 = model(test_input)

    print(f'任务1预测结果: {pred_task1}')
    print(f'任务2预测结果: {pred_task2}')

