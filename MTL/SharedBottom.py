import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SharedBottomMultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_task1_dim, output_task2_dim):
        super(SharedBottomMultiTaskModel, self).__init__()
        # 定义共享底部的三层全连接层
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.ReLU()
        )

        # 定义任务1的三层全连接层
        self.task1_head = nn.Sequential(
            nn.Linear(hidden3_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_task1_dim)
        )

        # 定义任务2的三层全连接层
        self.task2_head = nn.Sequential(
            nn.Linear(hidden3_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_task2_dim)
        )

    def forward(self, x):
        # 计算输入数据通过共享底部后的输出
        shared_output = self.shared_bottom(x)

        # 从共享底部输出分别计算任务1和任务2的结果
        task1_output = self.task1_head(shared_output)
        task1_output[:, 0] = torch.sigmoid(task1_output[:, 0])
        task1_output[:, 2] = torch.sigmoid(task1_output[:, 2])

        task2_output = self.task2_head(shared_output)
        task2_output[:, 1] = torch.sigmoid(task2_output[:, 1])

        # 返回任务1和任务2的结果
        return task1_output, task2_output


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
print('len(train_loader) = ', len(train_loader))   # num_samples=1000 / bs=32 再向上取整 =32


# 实例化模型对象
model = SharedBottomMultiTaskModel(input_dim, 64, 32, 16, task1_dim, task2_dim)

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