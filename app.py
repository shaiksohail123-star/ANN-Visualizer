import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_moons

st.title("Interactive ANN Visualizer")

# Sidebar controls
st.sidebar.header("Network Configuration")

input_nodes = st.sidebar.slider("Input Nodes", 2, 5, 2)
hidden_nodes = st.sidebar.slider("Hidden Nodes", 1, 10, 5)
output_nodes = st.sidebar.slider("Output Nodes", 1, 2, 1)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)

# Dataset
X, y = make_moons(n_samples=200, noise=0.2)
X = torch.tensor(X).float()
y = torch.tensor(y.reshape(-1,1)).float()

# Neural network model
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = ANN()

# Visualization: network structure
def draw_network():
    G = nx.DiGraph()

    inputs = [f"I{i}" for i in range(input_nodes)]
    hidden = [f"H{i}" for i in range(hidden_nodes)]
    outputs = [f"O{i}" for i in range(output_nodes)]

    G.add_nodes_from(inputs)
    G.add_nodes_from(hidden)
    G.add_nodes_from(outputs)

    for i in inputs:
        for h in hidden:
            G.add_edge(i, h)

    for h in hidden:
        for o in outputs:
            G.add_edge(h, o)

    pos = nx.spring_layout(G)

    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color="skyblue", ax=ax)

    st.pyplot(fig)

st.subheader("Neural Network Structure")
draw_network()

# Train model
if st.button("Train Model"):

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    st.success("Training Complete")

    # Loss graph
    st.subheader("Training Loss")
    st.line_chart(loss_history)

    # Decision boundary
    st.subheader("Decision Boundary")

    xx, yy = np.meshgrid(
        np.linspace(-2,3,100),
        np.linspace(-1.5,2,100)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid).float()

    preds = model(grid_tensor).detach().numpy()
    preds = preds.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, preds, cmap="coolwarm", alpha=0.6)
    ax.scatter(X[:,0], X[:,1], c=y.numpy(), cmap="coolwarm")

    st.pyplot(fig)
