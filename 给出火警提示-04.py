import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import os

# 定义文件路径
file_path = os.getenv('DATA_PATH', '.')
fire_data_file = os.path.join(file_path, 'california_fire_data.xlsx')

try:
    # 读取模拟数据文件（Excel格式）
    fire_data = pd.read_excel(fire_data_file)

    # 检查数据完整性
    required_columns = ['temperature', 'humidity', 'wind_speed', 'fuel_type', 'fire_risk']
    if not all(col in fire_data.columns for col in required_columns):
        raise ValueError("数据集中缺少必要的列")

    # 对字符串特征进行编码
    label_encoder = LabelEncoder()
    label_encoder.fit(fire_data['fuel_type'])
    fire_data['fuel_type'] = label_encoder.transform(fire_data['fuel_type'])

    # 数据标准化
    scaler = StandardScaler()
    features = fire_data[['temperature', 'humidity', 'wind_speed']]
    features_scaled = scaler.fit_transform(features)

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


# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
condition_dim = 1
output_dim = features_scaled.shape[1]
generator = Generator(z_dim, condition_dim, output_dim).to(device)

# 加载生成器权重
generator.load_state_dict(torch.load('generator_model.pth'))
generator.eval()

# 生成新的数据
num_samples = 1000
z = torch.randn(num_samples, z_dim).to(device)
conditions = torch.tensor(fire_data['fuel_type'].values[:num_samples].reshape(-1, 1), dtype=torch.float32).to(device)
generated_data = generator(z, conditions).cpu().detach().numpy()

# 反标准化生成的数据
generated_data = scaler.inverse_transform(generated_data)

# 将生成的数据转换为DataFrame
generated_df = pd.DataFrame(generated_data, columns=['temperature', 'humidity', 'wind_speed'])

# 添加fuel_type列
generated_df['fuel_type'] = fire_data['fuel_type'].values[:num_samples]

# 打印生成的数据
print(generated_df.head())

# 定义火警阈值
# 假设我们定义一个简单的阈值规则来判断是否需要发出火警提示
# 例如，如果温度超过35度且湿度低于30%，则发出火警提示
temperature_threshold = 35
humidity_threshold = 30

# 生成火警提示
fire_alerts = generated_df[(generated_df['temperature'] > temperature_threshold) & (generated_df['humidity'] < humidity_threshold)]

# 打印火警提示
if not fire_alerts.empty:
    print("\n火警提示:")
    print(fire_alerts)
else:
    print("\n当前没有火警风险。")
