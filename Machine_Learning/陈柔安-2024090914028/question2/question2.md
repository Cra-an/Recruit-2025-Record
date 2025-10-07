## 一、任务概述
1. 定义LeNet卷积神经网络结构
2. 加载并预处理MNIST数据集
3. 训练LeNet模型
4. 在测试集上评估模型准确率


## 二、详细解题步骤

### （一）定义LeNet卷积神经网络
1. **网络结构解析**  

   ```python
   class LeNet(nn.Module):
       def __init__(self):
           super(LeNet, self).__init__()
           # 卷积层1：输入1通道（灰度图），输出6通道，卷积核5×5
           self.conv1 = nn.Conv2d(1, 6, 5)
           self.relu = nn.ReLU()  # 激活函数，引入非线性
           self.pool1 = nn.MaxPool2d(2, 2)  # 2×2最大池化，步长2
           
           # 卷积层2：输入6通道，输出16通道，卷积核5×5
           self.conv2 = nn.Conv2d(6, 16, 5)
           self.pool2 = nn.MaxPool2d(2, 2)  # 2×2最大池化，步长2
           
           # 全连接层：将卷积特征映射到10个类别（0-9）
           self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 输入维度=16特征图×4×4大小
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 10)
   ```

2. **前向传播过程**  

   定义数据在网络中的流动路径 :  

```python
def forward(self, x):
    # 卷积1 → 激活 → 池化1
    x = self.pool1(self.relu(self.conv1(x)))
    # 卷积2 → 激活 → 池化2
    x = self.pool2(self.relu(self.conv2(x)))
    # 展平特征图为一维向量（-1表示自动计算批量大小）
    x = x.view(-1, 16 * 4 * 4)
    # 全连接层+激活
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    # 输出层（无激活，直接用于计算交叉熵损失）
    x = self.fc3(x)
    return x
```



### （二）加载与预处理 MNIST 数据集

1. **数据预处理管道**

   使用`transforms`对图像进行标准化处理：

   ```python
   transform = transforms.Compose([
       transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
       transforms.Normalize((0.1307,), (0.3081,))  # 用MNIST均值和标准差标准化
   ])
   ```

   标准化原因：使数据分布更稳定，加速模型收敛。

2. **数据集与数据加载器**  

   ```python
   # 加载训练集和测试集（自动下载至./data目录）
   train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
   
   # 批量加载数据（训练集打乱顺序，测试集保持顺序）
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
   ```

   - `batch_size=64`：每次训练 / 测试使用 64 个样本，平衡效率与内存。
   - `shuffle=True`：训练时打乱数据，避免模型学习数据顺序特征。

### （三）模型训练过程

1. **初始化训练组件**  

   ```python
   model = LeNet()  # 实例化模型
   criterion = nn.CrossEntropyLoss()  # 交叉熵损失
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD优化器
   epochs = 5  # 训练轮数
   ```

   优化器参数：`lr=0.01`为学习率，`momentum=0.9`引入动量加速收敛。

2. **迭代训练逻辑**  

   ```python
   for epoch in range(epochs):
       running_loss = 0.0  # 累计损失
       for i, (inputs, labels) in enumerate(train_loader):
           # 1. 清空梯度
           optimizer.zero_grad()
           
           # 2. 前向传播：计算预测输出
           outputs = model(inputs)
           
           # 3. 计算损失
           loss = criterion(outputs, labels)
           
           # 4. 反向传播：计算参数梯度
           loss.backward()
           
           # 5. 更新参数
           optimizer.step()
           
           # 监控损失（每100批次打印一次）
           running_loss += loss.item()
           if i % 100 == 99:
               print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.3f}')
               running_loss = 0.0
   ```

   核心流程：清空梯度→前向传播→计算损失→反向传播→更新参数，是深度学习训练的标准范式。

### （四）模型测试与准确率计算

1. **测试阶段设置**

   使用`torch.no_grad()`禁用梯度计算（测试无需更新参数，节省资源）：

   ```python
   correct = 0  # 正确预测数
   total = 0    # 总样本数
   with torch.no_grad():
       for inputs, labels in test_loader:
           outputs = model(inputs)  # 模型预测
           # 获取预测概率最大的类别（dim=1表示按行取最大值）
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)  # 累加总样本数
           correct += (predicted == labels).sum().item()  # 累加正确数
   ```

2. **计算准确率**  

   ```python
   accuracy = 100 * correct / total
   print(f'测试集上的准确率: {accuracy:.2f}%')
   ```

## 三、关键代码解析

| 代码片段                     | 作用说明                                           |
| ---------------------------- | -------------------------------------------------- |
| `nn.Conv2d(1, 6, 5)`         | 卷积层：通过 5×5 卷积核从 1 通道输入提取 6 种特征  |
| `nn.MaxPool2d(2, 2)`         | 池化层：将特征图尺寸缩小一半，保留关键特征         |
| `x.view(-1, 16*4*4)`         | 特征展平：将二维特征图转换为一维向量，适配全连接层 |
| `loss.backward()`            | 反向传播：自动计算各参数的梯度，用于参数更新       |
| `torch.max(outputs.data, 1)` | 预测类别：取输出向量中概率最大的索引作为预测结果   |

## 四、结果分析

![](..\assets\question2\result1.png)

![](..\assets\question2\result2.png)

- **训练趋势**：随着训练轮次增加，损失值应逐渐降低并趋于稳定，说明模型有效学习了数据特征。
- **测试准确率**：在 MNIST 数据集上，LeNet 模型通常可达到 98% 以上的准确率，验证了卷积神经网络对图像特征的提取能力。
- **优化方向**：可通过调整学习率、增加训练轮数、使用数据增强（如随机旋转）或更换优化器（如 Adam）进一步提升性能。