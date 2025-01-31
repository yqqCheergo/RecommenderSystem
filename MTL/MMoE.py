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

class MMoEMultiTaskModel(nn.Module):
    def __init__(self, input_size, num_experts, experts_out, experts_hidden_list, towers_hidden, tasks):
        super(MMoEMultiTaskModel, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden_list = experts_hidden_list
        self.towers_hidden = towers_hidden
        self.tasks = tasks

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # expert每个网络的输入特征都是一样的，其网络结构也是一致的
        self.experts = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden_list) for i in range(self.num_experts)])

        '''
        gate网络的数量与task数量相同
        gate网络最后一层全连接层的隐藏单元（即输出）size必须等于expert个数
        多个gate网络的输入是一样的，网络结构也是一致的
        '''
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True) for i in range(self.tasks)])

        # tower层的输入size等于expert输出的隐藏单元个数
        self.towers = nn.ModuleList([Tower(self.experts_out, task_dim[i], self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        experts_output = [expert(x) for expert in self.experts]
        # print('len(experts_output): ', len(experts_output))   # 3
        # print('experts_output[0].shape: ', experts_output[0].shape)  # torch.Size([32, 16])
        experts_output_tensor = torch.stack(experts_output)
        # print('experts_output_tensor.shape: ', experts_output_tensor.shape)    # torch.Size([3, 32, 16])


        # gate网络最后一层全连接层要经过softmax归一化
        '''
        x: torch.Size([32, 10])
        gate: torch.Size([10, 3])
        @为矩阵乘法
        '''
        gates_output = [self.softmax(x @ gate) for gate in self.w_gates]
        # print('len(gates_output): ', len(gates_output))   # 2
        # print('gates_output[0].shape: ', gates_output[0].shape)  # torch.Size([32, 3])
        # print('gates_output[0].t().unsqueeze(2).shape: ', gates_output[0].t().unsqueeze(2).shape)  # torch.Size([3, 32, 1])
        # print('gates_output[0].t().unsqueeze(2).expand(-1, -1, self.experts_out).shape: ', gates_output[0].t().unsqueeze(2).expand(-1, -1, self.experts_out).shape)  # torch.Size([3, 32, 16])


        '''
        *左侧：torch.Size([3, 32, 16])
        *右侧：torch.Size([3, 32, 16])
        *为逐元素相乘
        '''
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_output_tensor for g in gates_output]
        # print('len(tower_input): ', len(tower_input))    # 2
        # print('tower_input[0].shape: ', tower_input[0].shape)   # torch.Size([3, 32, 16])
        # print('torch.sum(tower_input[0], dim=0).shape: ', torch.sum(tower_input[0], dim=0).shape)   # torch.Size([32, 16])


        # 3个Expert的输出加权求和，输入特定的任务塔中
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        # print("len(tower_input): ", len(tower_input))    # 2
        # print("tower_input[0].shape: ", tower_input[0].shape)   # torch.Size([32, 16])


        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        # print('len(final_output): ', len(final_output))   # 2
        task1_output, task2_output = final_output[0], final_output[1]
        # print('task1_output.shape: ', task1_output.shape)   # torch.Size([32, 3])
        # print('task2_output.shape: ', task2_output.shape)   # torch.Size([32, 2])
        task1_output[:, 0] = self.sigmoid(task1_output[:, 0])
        task1_output[:, 2] = self.sigmoid(task1_output[:, 2])
        task2_output[:, 1] = self.sigmoid(task2_output[:, 1])

        return task1_output, task2_output


# 构造虚拟样本数据
torch.manual_seed(42)  # 设置随机种子以保证结果可重复
input_dim = 10
task1_dim = 3
task2_dim = 2
task_dim = [task1_dim, task2_dim]
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
model = MMoEMultiTaskModel(input_dim, 3, 16, experts_hidden_list, 32, 2)

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