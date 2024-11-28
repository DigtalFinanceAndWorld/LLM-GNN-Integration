from torch_geometric.nn import SAGEConv
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_process import load_gnn_data


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x are the node features and edge_index are the edges of the graph
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def run_GraphSAGE(dataset_dir, embeddings_csv_dir, index, in_channels, hidden_channels, out_channels, learning_rate, num_epochs):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, test_data = load_gnn_data(dataset_dir, embeddings_csv_dir, index, device)

    # 模型定义：Model, Loss, Optimizer
    model = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)

    def train():
        for epoch in range(num_epochs):
            # 共进行200次训练
            model.train()
            out = model(train_data.x, train_data.edge_index)  # 前向传播
            loss = F.mse_loss(out, train_data.x)

            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            # 验证集
            model.eval()
            with torch.no_grad():
                out = model(val_data.x, val_data.edge_index)
                #             reconstructed_features = decoder(out)  # 解码器的输出
                # 然后可以计算重建特征与原始特征之间的损失
                loss1 = F.mse_loss(out, val_data.x)

            print("Epoch {:03d}: train_Loss {:.4f}: val_Loss {:.4f}".format(
                epoch, loss.item(), loss1.item()))



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
    train_df.to_csv(f'{embeddings_csv_dir}/{index}_train_embeddings.csv', index=False)
    val_df.to_csv(f'{embeddings_csv_dir}/{index}_val_embeddings.csv', index=False)
    test_df.to_csv(f'{embeddings_csv_dir}/{index}_test_embeddings.csv', index=False)
    torch.cuda.empty_cache()