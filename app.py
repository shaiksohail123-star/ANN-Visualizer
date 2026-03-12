import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.datasets import make_moons
import numpy as np

st.title("Interactive ANN Visualizer")

# Sidebar controls
st.sidebar.header("Network Configuration")

input_nodes = st.sidebar.slider("Input Nodes", 1, 10, 2)
hidden_nodes = st.sidebar.slider("Hidden Nodes", 1, 20, 5)
output_nodes = st.sidebar.slider("Output Nodes", 1, 5, 1)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)

# Neural Network
class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = ANN()

# Dataset
X, y = make_moons(n_samples=200, noise=0.2)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

# Training
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

losses = []

if st.button("Train Model"):

    for epoch in range(epochs):

        optimizer.zero_grad()

        output = model(X)

        loss = loss_fn(output, y)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    st.success("Training Completed")

# Loss Graph
if len(losses) > 0:

    fig, ax = plt.subplots()

    ax.plot(losses)

    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    st.pyplot(fig)

# Neural Network Diagram
st.subheader("Neural Network Structure")

G = nx.DiGraph()

for i in range(input_nodes):
    G.add_node(f"I{i}")

for i in range(hidden_nodes):
    G.add_node(f"H{i}")

for i in range(output_nodes):
    G.add_node(f"O{i}")

pos = {}

for i in range(input_nodes):
    pos[f"I{i}"] = (0, i)

for i in range(hidden_nodes):
    pos[f"H{i}"] = (1, i)

for i in range(output_nodes):
    pos[f"O{i}"] = (2, i)

fig, ax = plt.subplots()

nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500)

st.pyplot(fig)

# Weight Heatmap
st.subheader("Weight Heatmap")

weights = model.fc1.weight.detach().numpy()

fig, ax = plt.subplots()

sns.heatmap(weights, cmap="coolwarm", ax=ax)

st.pyplot(fig)