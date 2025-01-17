import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

# 定义文件路径
file_path = os.getenv('DATA_PATH', '.')
fire_data_file = os.path.join(file_path, 'california_fire_data.xlsx')
satellite_data_file = os.path.join(file_path, 'real_time_satellite_data.xlsx')

try:
    # 读取模拟数据文件（Excel格式）
    fire_data = pd.read_excel(fire_data_file)
    real_time_satellite_data = pd.read_excel(satellite_data_file)

    # 检查数据完整性
    required_columns = ['temperature', 'humidity', 'wind_speed', 'fuel_type', 'fire_risk']
    if not all(col in fire_data.columns for col in required_columns) or not all(col in real_time_satellite_data.columns for col in required_columns[:-1]):
        raise ValueError("数据集中缺少必要的列")

    # 对字符串特征进行编码
    label_encoder = LabelEncoder()
    label_encoder.fit(fire_data['fuel_type'])
    fire_data['fuel_type'] = label_encoder.transform(fire_data['fuel_type'])
    real_time_satellite_data['fuel_type'] = label_encoder.transform(real_time_satellite_data['fuel_type'])

    # 数据标准化
    scaler = StandardScaler()
    features = fire_data[['temperature', 'humidity', 'wind_speed']]
    features_scaled = scaler.fit_transform(features)

    real_time_features = real_time_satellite_data[['temperature', 'humidity', 'wind_speed']]
    real_time_features_scaled = scaler.transform(real_time_features)

except Exception as e:
    print(f"数据读取或处理失败: {e}")
    exit(1)

# 定义cWGAN模型
class Generator(nn.Module):
    def __init__(self, z_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim + condition_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class Critic(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Critic, self).__init__()
        # 输入维度为input_dim + condition_dim
        self.fc1 = nn.Linear(input_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# 计算梯度惩罚
def compute_gradient_penalty(critic, real_samples, fake_samples, conditions, device):
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, conditions)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=False,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 训练cWGAN模型
def train_cWGAN(generator, critic, data_loader, n_epochs, device):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

    for epoch in range(n_epochs):
        for real_data, conditions in data_loader:  # 解包两个元素
            real_data = real_data.to(device)
            conditions = conditions.to(device)

            # Train Critic
            optimizer_C.zero_grad()
            z = torch.randn(real_data.size(0), 100).to(device)
            fake_data = generator(z, conditions).detach()
            real_validity = critic(real_data, conditions).mean()
            fake_validity = critic(fake_data, conditions).mean()
            gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data, conditions, device)
            c_loss = fake_validity - real_validity + 10 * gradient_penalty
            optimizer_C.step()

            # Train Generator
            if epoch % 5 == 0:
                optimizer_G.zero_grad()
                z = torch.randn(real_data.size(0), 100).to(device)
                fake_data = generator(z, conditions)
                g_loss = -critic(fake_data, conditions).mean()
                g_loss.backward()  # 不需要 retain_graph=True
                optimizer_G.step()

        print(f"Epoch [{epoch}/{n_epochs}], C Loss: {c_loss.item()}, G Loss: {g_loss.item()}")


# 准备训练数据
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
conditions_tensor = torch.tensor(fire_data['fuel_type'].values.reshape(-1, 1), dtype=torch.float32)
dataset = TensorDataset(features_tensor, conditions_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim=100, condition_dim=1, output_dim=features_scaled.shape[1]).to(device)
critic = Critic(input_dim=features_scaled.shape[1], condition_dim=1).to(device)

# 训练模型
n_epochs = 100
train_cWGAN(generator, critic, data_loader, n_epochs, device)

# 保存生成器模型
torch.save(generator.state_dict(), 'generator_model.pth')
# 保存判别器模型
torch.save(critic.state_dict(), 'critic_model.pth')