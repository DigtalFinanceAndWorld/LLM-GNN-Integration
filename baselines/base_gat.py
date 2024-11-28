import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from data_process import load_gnn_data
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


class GAT(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim, output_dim, heads=4, edge_dim=2):
        super().__init__()
        self.gat1 = GATv2Conv(input_feat_dim, hidden_dim, heads=heads, edge_dim=2)
        self.gat2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1, edge_dim=2)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)
        return x


def train(model, train_data, optimizer, val_data, num_epochs):
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 使用混合精度训练
        with autocast():
            out = model(x=train_data.x, edge_attr=train_data.edge_attr, edge_index=train_data.edge_index)
            loss = F.mse_loss(out, train_data.x)

        # 使用梯度缩放器进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_history.append(loss.item())

        # 验证集
        model.eval()
        with torch.no_grad():
            out = model(x=val_data.x, edge_attr=val_data.edge_attr, edge_index=val_data.edge_index)
            # 然后可以计算重建特征与原始特征之间的损失
            loss1 = F.mse_loss(out, val_data.x)

        print("Epoch {:03d}: train_Loss {:.4f}: val_Loss {:.4f}".format(
            epoch, loss.item(), loss1.item()))

    return loss_history

    # 测试函数


def test(model, train_data, val_data, test_data):
    model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

    with torch.no_grad():  # 显著减少显存占用
        embeddings = model(x=train_data.x, edge_attr=train_data.edge_attr, edge_index=train_data.edge_index)
        embeddings1 = model(x=val_data.x, edge_attr=val_data.edge_attr, edge_index=val_data.edge_index)
        embeddings2 = model(x=test_data.x, edge_attr=test_data.edge_attr, edge_index=test_data.edge_index)

    return embeddings.cpu().numpy(), embeddings1.cpu().numpy(), embeddings2.cpu().numpy()


def run_GAT(dataset_dir, embeddings_csv_dir, index, input_feat_dim, hidden_dim, output_dim, learning_rate, num_epochs):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, test_data = load_gnn_data(dataset_dir, embeddings_csv_dir, index, device)

    model = GAT(input_feat_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)
    loss = train(model, train_data, optimizer, val_data, num_epochs)

    train_embedding, val_embedding, test_embedding = test(model, train_data, val_data, test_data)
    train_df = pd.DataFrame(train_embedding)
    val_df = pd.DataFrame(val_embedding)
    test_df = pd.DataFrame(test_embedding)
    train_df.to_csv(f'{embeddings_csv_dir}/{index}_train_embeddings.csv', index=False)
    val_df.to_csv(f'{embeddings_csv_dir}/{index}_val_embeddings.csv', index=False)
    test_df.to_csv(f'{embeddings_csv_dir}/{index}_test_embeddings.csv', index=False)
    torch.cuda.empty_cache()
