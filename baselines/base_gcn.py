import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from data_process import load_gnn_data


class GcnNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=8):
        super(GcnNet, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, output_dim)

    def forward(self, feature, edge_index):
        h1 = F.relu(self.gcn1(feature, edge_index))
        h2 = F.relu(self.gcn2(h1, edge_index))
        logits = self.gcn3(h2, edge_index)
        return logits


def run_GCN(dataset_dir, embeddings_csv_dir, index, input_feat_dim, hidden_dim, output_dim, learning_rate, num_epochs):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, test_data = load_gnn_data(dataset_dir, embeddings_csv_dir, index, device)

    # 模型定义：Model, Loss, Optimizer
    model = GcnNet(input_feat_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)

    def train():

        for epoch in range(num_epochs):
            model.train()
            out = model(train_data.x, train_data.edge_index)  # 前向传播
            loss = F.mse_loss(out, train_data.x)

            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            # loss_history.append(loss.item())

            # 验证集
            model.eval()
            with torch.no_grad():
                out = model(val_data.x, val_data.edge_index)
                #             reconstructed_features = decoder(out)  # 解码器的输出
                # 然后可以计算重建特征与原始特征之间的损失
                loss1 = F.mse_loss(out, val_data.x)

            print("Epoch {:03d}: train_Loss {:.4f}: val_Loss {:.4f}".format(
                epoch, loss.item(), loss1.item()))

    # 测试函数
    def test():
        model.eval()  # 表示将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

        with torch.no_grad():  # 显著减少显存占用
            train_embeddings = model(train_data.x, train_data.edge_index)  # (N,16)->(N,7) N节点数
            val_embeddings = model(val_data.x, val_data.edge_index)
            test_embeddings = model(test_data.x, test_data.edge_index)

        return train_embeddings.cpu().numpy(), val_embeddings.cpu().numpy(), test_embeddings.cpu().numpy()

    train()
    train_embedding, val_embedding, test_embedding = test()
    train_df = pd.DataFrame(train_embedding)
    val_df = pd.DataFrame(val_embedding)
    test_df = pd.DataFrame(test_embedding)
    print(f"train_df: {len(train_df)}")
    train_df.to_csv(f'{embeddings_csv_dir}/{index}_train_embeddings.csv', index=False)
    val_df.to_csv(f'{embeddings_csv_dir}/{index}_val_embeddings.csv', index=False)
    test_df.to_csv(f'{embeddings_csv_dir}/{index}_test_embeddings.csv', index=False)
    torch.cuda.empty_cache()
