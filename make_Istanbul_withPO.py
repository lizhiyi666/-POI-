import os
import torch
import numpy as np
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# -------------------------- 1. 全局配置 --------------------------
PO_ENCODING_DIM = 32  # 偏序编码后的固定维度（可根据N调整）
DATA_ROOT = "data/Istanbul"
OUTPUT_ROOT = "data/Istanbul_PO1"


# -------------------------- 2. 工具函数 --------------------------
def collect_all_categories(train_data, test_data):
    """收集所有POI类别，建立固定映射（与之前一致）"""
    all_categories = set()
    for seq in train_data['sequences']:
        all_categories.update(seq['marks'])
    for seq in test_data['sequences']:
        all_categories.update(seq['marks'])
    categories = sorted(list(all_categories))
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    return categories, cat2idx


def get_full_partial_order_matrix(seq_marks, cat2idx):
    """生成完整的N×N偏序邻接矩阵（保留所有偏序关系）"""
    num_cats = len(cat2idx)
    adj_matrix = np.zeros((num_cats, num_cats), dtype=np.float32)

    # 记录每个类别的首次/末次出现位置
    first_occur = defaultdict(lambda: float('inf'))
    last_occur = defaultdict(lambda: -float('inf'))
    for idx, cat in enumerate(seq_marks):
        cat_idx = cat2idx[cat]
        first_occur[cat_idx] = min(first_occur[cat_idx], idx)
        last_occur[cat_idx] = max(last_occur[cat_idx], idx)

    # 遍历所有类别对，判断偏序关系
    for a in range(num_cats):
        for b in range(num_cats):
            if a == b:
                continue
            # 若a的末次出现 < b的首次出现 → a→b（偏序关系）
            if last_occur[a] < first_occur[b] and first_occur[b] != float('inf'):
                adj_matrix[a][b] = 1.0
    return adj_matrix


def fit_svd_on_all_po_matrices(train_data, test_data, cat2idx):
    """基于全量偏序矩阵训练SVD模型（保证编码一致性）"""
    all_po_matrices = []
    num_cats = len(cat2idx)

    # 收集训练集+测试集的所有偏序矩阵
    for data in [train_data, test_data]:
        for seq in data['sequences']:
            po_matrix = get_full_partial_order_matrix(seq['marks'], cat2idx)
            all_po_matrices.append(po_matrix.reshape(-1))  # 展平为1×N²向量

    # 标准化 + 训练SVD
    scaler = StandardScaler()
    all_po_matrices_scaled = scaler.fit_transform(np.array(all_po_matrices))
    svd = TruncatedSVD(n_components=PO_ENCODING_DIM, random_state=135398)
    svd.fit(all_po_matrices_scaled)

    return scaler, svd


def encode_po_matrix(po_matrix, scaler, svd):
    """将单个偏序矩阵编码为低维向量"""
    # 展平 → 标准化 → SVD降维
    po_flat = po_matrix.reshape(1, -1)
    po_scaled = scaler.transform(po_flat)
    po_encoded = svd.transform(po_scaled).reshape(-1)  # 输出维度：PO_ENCODING_DIM
    return po_encoded.astype(np.float32)


# -------------------------- 3. 数据处理主逻辑 --------------------------
def process_dataset(original_data, cat2idx, scaler, svd):
    """为每个序列添加完整偏序矩阵 + 低维编码向量"""
    new_sequences = []
    num_cats = len(cat2idx)

    for seq in original_data['sequences']:
        # 生成完整偏序矩阵
        po_matrix = get_full_partial_order_matrix(seq['marks'], cat2idx)
        # 编码为低维向量
        po_encoded = encode_po_matrix(po_matrix, scaler, svd)

        # 复制原始序列，新增2个字段（不修改任何原有字段）
        new_seq = seq.copy()
        new_seq['po_matrix'] = po_matrix  # 保留完整偏序矩阵（可选）
        new_seq['po_encoding'] = po_encoded  # 核心：偏序低维编码向量
        new_sequences.append(new_seq)

    # 构建新数据集（保留原有所有字段 + 新增全局信息）
    new_data = {
        **original_data,
        'sequences': new_sequences,
        'category_mapping': cat2idx,
        'po_encoding_dim': PO_ENCODING_DIM,
        'num_categories': num_cats
    }
    return new_data


# -------------------------- 4. 执行流程 --------------------------
def main():
    # 1. 加载原始数据
    train_path = os.path.join(DATA_ROOT, 'Istanbul_train.pkl')
    test_path = os.path.join(DATA_ROOT, 'Istanbul_test.pkl')
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)

    # 2. 收集类别映射
    categories, cat2idx = collect_all_categories(train_data, test_data)
    print(f"总POI类别数：{len(cat2idx)}，偏序编码维度：{PO_ENCODING_DIM}")

    # 3. 训练SVD模型（基于全量偏序矩阵）
    scaler, svd = fit_svd_on_all_po_matrices(train_data, test_data, cat2idx)
    print(f"SVD解释方差比：{sum(svd.explained_variance_ratio_):.4f}")

    # 4. 处理训练/测试集
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    new_train = process_dataset(train_data, cat2idx, scaler, svd)
    new_test = process_dataset(test_data, cat2idx, scaler, svd)

    # 5. 保存新数据集（不修改原有字段，仅新增）
    torch.save(new_train, os.path.join(OUTPUT_ROOT, 'Istanbul_PO1_train.pkl'))
    torch.save(new_test, os.path.join(OUTPUT_ROOT, 'Istanbul_PO1_test.pkl'))
    print(f"新数据集已保存至 {OUTPUT_ROOT}")

    # 验证：打印第一个序列的编码结果
    sample_seq = new_train['sequences'][0]
    print(f"示例序列偏序编码向量形状：{sample_seq['po_encoding'].shape}")
    print(f"示例序列偏序编码向量前5值：{sample_seq['po_encoding'][:5]}")


if __name__ == "__main__":
    main()