# Import necessary libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from scipy.spatial.distance import euclidean
from google.colab import files
import zipfile

# Step 1: Upload & Extract Dataset

uploaded = files.upload()  # Prompts user to upload a file

# Extract ZIP File
for filename in uploaded.keys():
    if filename.endswith(".zip"):
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall("/content/dataset")  # Extract to /content/dataset
        zip_ref.close()
        os.remove(filename)  # Remove ZIP file after extraction

data_dir = "/content/dataset"  # Dataset path

# ==========================
# ✅ Step 2: Validate Dataset

valid_extensions = {".jpg", ".jpeg", ".png"}

# Check if dataset exists & contains valid images
if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
    raise FileNotFoundError(f"Dataset not found in {data_dir}. Please upload a valid dataset.")

for root, _, files in os.walk(data_dir):
    for file in files:
        if not file.lower().endswith(tuple(valid_extensions)):
            raise ValueError(f"Invalid file found: {file}. Expected image files (.jpg, .jpeg, .png).")

# Step 3: Load Dataset for Training

# Image transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ✅ Step 4: Define Deep Q-Network (DQN)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten image input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 5: Firefly Algorithm for Optimization

def firefly_algorithm(n_fireflies, max_iter, obj_func, dim):
    alpha = 0.5  # Randomness strength
    beta_0 = 1.0  # Attraction factor
    gamma = 1.0  # Absorption coefficient

    fireflies = np.random.rand(n_fireflies, dim)
    best_solution = fireflies[0]
    best_score = float('inf')

    for _ in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if obj_func(fireflies[j]) < obj_func(fireflies[i]):
                    r = euclidean(fireflies[i], fireflies[j])
                    beta = beta_0 * np.exp(-gamma * r ** 2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(dim) - 0.5)
                    fireflies[i] = np.clip(fireflies[i], 0, 1)

                    if obj_func(fireflies[i]) < best_score:
                        best_score = obj_func(fireflies[i])
                        best_solution = fireflies[i]

    return best_solution

# Step 6: Define Custom Facial Recognition RL Environment

class FacialRecEnv(gym.Env):
    def __init__(self):
        super(FacialRecEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(64, 64), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)  # Accept (1) or Reject (0)

    def reset(self):
        return np.random.rand(64, 64)

    def step(self, action):
        reward = 1 if action == np.random.choice([0, 1]) else -1  # Random reward logic
        return np.random.rand(64, 64), reward, False, {}

# 🚀 Step 7: Train DQN with Firefly Algorithm

def train_dqn():
    env = FacialRecEnv()
    dqn = DQN(64 * 64, env.action_space.n)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    gamma = 0.9  # Discount factor

    memory = []  # Replay memory buffer

    for episode in range(50):  # Train for 50 episodes
        state = env.reset()
        state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        done = False

        while not done:
            q_values = dqn(state)
            action = torch.argmax(q_values).item()
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float32).unsqueeze(0)

            memory.append((state, action, reward, next_state))

            # Train the model if memory has enough samples
            if len(memory) > 10:
                batch = np.random.choice(memory, 10, replace=False)
                batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

                batch_states = torch.cat(batch_states)
                batch_next_states = torch.cat(batch_next_states)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)

                target_qs = batch_rewards + gamma * torch.max(dqn(batch_next_states), dim=1)[0]
                predicted_qs = dqn(batch_states).gather(1, torch.tensor(batch_actions).unsqueeze(1)).squeeze()

                loss = loss_fn(predicted_qs, target_qs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Optimize DQN weights using Firefly Algorithm
        def obj_func(weights):
            with torch.no_grad():
                dqn.fc1.weight.data = torch.tensor(weights[:128].reshape(128, -1))
                dqn.fc2.weight.data = torch.tensor(weights[128:256].reshape(128, -1))
                dqn.fc3.weight.data = torch.tensor(weights[256:].reshape(2, -1))
                return -torch.mean(dqn(state)).item()

        best_weights = firefly_algorithm(10, 20, obj_func, 258)
        print(f"Episode {episode}: Firefly Algorithm optimized the Q-values.")

    torch.save(dqn.state_dict(), "dqn_model.pth")
    print("Model saved as dqn_model.pth")

# ✅ Step 8: Run Training
train_dqn()


