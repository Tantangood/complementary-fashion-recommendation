import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from transformers import BertModel

class DynamicConvEncoder(nn.Module):
    def __init__(self, input_dim, num_conditions, kernel_size=3):
        super(DynamicConvEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_conditions = num_conditions
        self.kernel_size = kernel_size
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, num_conditions * (input_dim * kernel_size + 1))
        )

    def forward(self, x):
        params = self.layers(x)
        weights = params[:, :self.num_conditions * self.input_dim * self.kernel_size]
        weights = weights.view(-1, self.num_conditions, self.input_dim, self.kernel_size)
        biases = params[:, self.num_conditions * self.input_dim * self.kernel_size:]
        biases = biases.view(-1, self.num_conditions, 1)
        return weights, biases

class C3Rec(nn.Module):
    def __init__(self, args):
        super(C3Rec, self).__init__()
        self.args = args
        self.embed_size = args.embed_size
        self.num_items = args.num_items
        self.num_conditions = 5
        self.alpha = args.alpha
        self.lambda_reg = args.lambda_reg
        self.r1 = args.r1
        self.outdim_size = args.outdim_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_fc = nn.Linear(512 + 256 + 128 + 64, self.embed_size)
        nn.init.xavier_uniform_(self.visual_fc.weight)
        nn.init.constant_(self.visual_fc.bias, 0)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, self.embed_size)
        nn.init.xavier_uniform_(self.text_fc.weight)
        nn.init.constant_(self.text_fc.bias, 0)

        self.semantic_conditions = nn.Parameter(torch.randn(self.num_conditions, self.embed_size * 2))
        nn.init.xavier_uniform_(self.semantic_conditions)

        self.dynamic_conv = DynamicConvEncoder(
            input_dim=self.embed_size * 2 * self.num_items,
            num_conditions=self.num_conditions
        )

        self.compat_mlp = nn.Sequential(
            nn.Linear(self.embed_size * 2 * self.num_items, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        for layer in self.compat_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        self.cnn_f1 = nn.Sequential(
            nn.Conv1d(self.embed_size * 2, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, self.embed_size, kernel_size=3, padding=1)
        )
        self.cnn_f2 = nn.Sequential(
            nn.Conv1d(self.embed_size * 2, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, self.embed_size, kernel_size=3, padding=1)
        )
        for cnn in [self.cnn_f1, self.cnn_f2]:
            for layer in cnn:
                if isinstance(layer, nn.Conv1d):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

        indices = torch.stack([
            torch.arange(self.num_items * self.embed_size * 2),
            torch.randperm(self.num_items * self.embed_size * 2)
        ])
        values = torch.ones(self.num_items * self.embed_size * 2)
        self.sparse_Q = nn.Parameter(
            torch.sparse_coo_tensor(
                indices, values, (self.num_items * self.embed_size * 2, self.num_items * self.embed_size * 2)
            ),
            requires_grad=True
        )

    def extract_visual_features(self, images):
        batch_size, item_num, C, H, W = images.shape
        images = images.view(-1, C, H, W)

        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        rep_l1 = self.resnet.layer1(x)
        rep_l2 = self.resnet.layer2(rep_l1)
        rep_l3 = self.resnet.layer3(rep_l2)
        rep_l4 = self.resnet.layer4(rep_l3)

        rep_l1 = self.global_avg_pool(rep_l1).squeeze(-1).squeeze(-1).view(batch_size, item_num, 64)
        rep_l2 = self.global_avg_pool(rep_l2).squeeze(-1).squeeze(-1).view(batch_size, item_num, 128)
        rep_l3 = self.global_avg_pool(rep_l3).squeeze(-1).squeeze(-1).view(batch_size, item_num, 256)
        rep_l4 = self.global_avg_pool(rep_l4).squeeze(-1).squeeze(-1).view(batch_size, item_num, 512)

        visual_features = torch.cat([rep_l1, rep_l2, rep_l3, rep_l4], dim=2)
        visual_features = self.visual_fc(visual_features)
        return visual_features

    def extract_text_features(self, input_ids, attention_mask):
        batch_size, item_num, max_len = input_ids.shape
        input_ids = input_ids.view(-1, max_len)
        attention_mask = attention_mask.view(-1, max_len)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        text_features = outputs[:, 0, :]  # CLS token
        text_features = self.text_fc(text_features)
        text_features = text_features.view(batch_size, item_num, self.embed_size)
        return text_features

    def fine_grained_compatibility(self, multimodal_features):
        batch_size, item_num, embed_size = multimodal_features.shape
        assert embed_size == self.embed_size * 2, "Multimodal feature size mismatch"

        compat_scores = []
        x = multimodal_features.view(batch_size, -1)
        weights, biases = self.dynamic_conv(x)

        for k in range(self.num_conditions):
            s_k = self.semantic_conditions[k:k+1]
            e_ik = multimodal_features * s_k

            e_ik = e_ik.transpose(1, 2).unsqueeze(-1)
            W_k = weights[:, k]
            b_k = biases[:, k]
            e_ik_tilde = F.conv2d(e_ik, W_k.unsqueeze(-2), b_k.unsqueeze(-1))
            e_ik_tilde = F.relu(e_ik_tilde.squeeze(-1).transpose(1, 2))

            s_k = self.compat_mlp(e_ik_tilde.view(batch_size, -1))
            compat_scores.append(s_k)

        p_j = torch.stack(compat_scores, dim=0).mean(dim=0)
        return p_j

    def multi_view_correlation(self, multimodal_features):
        batch_size, item_num, embed_size = multimodal_features.shape

        H = multimodal_features.view(batch_size, -1)
        G = torch.matmul(H, self.sparse_Q.to_dense())
        G = G.view(batch_size, item_num, embed_size)

        H_reshaped = H.view(batch_size, item_num, -1).transpose(1, 2)
        G_reshaped = G.transpose(1, 2)
        F_H = self.cnn_f1(H_reshaped).transpose(1, 2)
        F_G = self.cnn_f2(G_reshaped).transpose(1, 2)

        F_H_bar = F_H - F_H.mean(dim=1, keepdim=True)
        F_G_bar = F_G - F_G.mean(dim=1, keepdim=True)

        Sigma_HH = torch.matmul(F_H_bar.transpose(1, 2), F_H_bar) / (item_num - 1) + self.r1 * torch.eye(self.embed_size, device=self.device)
        Sigma_GG = torch.matmul(F_G_bar.transpose(1, 2), F_G_bar) / (item_num - 1) + self.r1 * torch.eye(self.embed_size, device=self.device)
        Sigma_HG = torch.matmul(F_H_bar.transpose(1, 2), F_G_bar) / (item_num - 1)

        def matrix_sqrt_inv(M):
            try:
                eigvals, eigvecs = torch.linalg.eigh(M)
                eigvals = torch.clamp(eigvals, min=1e-6)
                return torch.matmul(eigvecs, torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(1, 2))
            except RuntimeError:
                return torch.linalg.pinv(M).sqrt()

        Sigma_HH_inv_sqrt = matrix_sqrt_inv(Sigma_HH)
        Sigma_GG_inv_sqrt = matrix_sqrt_inv(Sigma_GG)

        T = torch.matmul(torch.matmul(Sigma_HH_inv_sqrt, Sigma_HG), Sigma_GG_inv_sqrt)
        U, S, _ = torch.svd(T + 1e-6 * torch.eye(T.size(1), device=self.device))
        q_j = S[:, :self.outdim_size].sum(dim=1) / self.outdim_size
        return q_j, T

    def forward(self, images, input_ids, attention_mask):
        visual_features = self.extract_visual_features(images)
        text_features = self.extract_text_features(input_ids, attention_mask)
        multimodal_features = torch.cat([visual_features, text_features], dim=2)

        p_j = self.fine_grained_compatibility(multimodal_features)
        p_j = torch.sigmoid(p_j)

        q_j, T = self.multi_view_correlation(multimodal_features)
        q_j = torch.sigmoid(q_j)

        zeta_j = (1 - self.alpha) * p_j.squeeze() + self.alpha * q_j
        zeta_j = torch.sigmoid(zeta_j)

        return zeta_j, p_j, q_j, T

    def compute_loss(self, zeta_j, p_j, q_j, T, labels):
        L_com = F.binary_cross_entropy(p_j.squeeze(), labels.float())
        S = torch.svd(T + 1e-6 * torch.eye(T.size(1), device=self.device))[1]
        L_corr = -S[:self.outdim_size].sum() / self.outdim_size
        L_reg = self.lambda_reg * sum(p.norm(2) for p in self.parameters() if p.requires_grad)
        L_total = L_com + L_corr + L_reg
        return L_total, L_com, L_corr

    def recommend_fitb(self, imgs_list, input_ids_list, attention_mask_list):
        scores = []
        for imgs, input_ids, attention_mask in zip(imgs_list, input_ids_list, attention_mask_list):
            imgs = imgs.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            zeta_j, _, _, _ = self.forward(imgs, input_ids, attention_mask)
            scores.append(zeta_j)
        scores = torch.stack(scores, dim=1)
        best_idx = scores.argmax(dim=1)
        return best_idx, scores
