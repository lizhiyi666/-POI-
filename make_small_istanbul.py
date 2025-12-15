import os
import random
import torch

def make_small_istanbul(root="data/Istanbul",
                        src_name="Istanbul_train.pkl",
                        dst_name="Istanbul_small_train.pkl",
                        frac=1.0,  # 使用多少比例的数据：0.1 = 10%
                        seed=42):
    random.seed(seed)

    src_path = os.path.join(root, src_name)
    dst_path = os.path.join(root, dst_name)

    print(f"Loading dataset from {src_path}")
    loader = torch.load(src_path, map_location="cpu")

    sequences = loader["sequences"]
    n_total = len(sequences)
    n_keep = max(1, int(n_total * frac))

    print(f"Total sequences: {n_total}")
    print(f"Keeping {n_keep} sequences (~{frac*100:.1f}%)")

    # 随机抽取一部分序列
    small_sequences = random.sample(sequences, n_keep)

    loader["sequences"] = small_sequences  # 其他 key 保持不变：t_max, num_marks, num_pois, poi_gps 等

    torch.save(loader, dst_path)
    print(f"Saved small dataset to {dst_path}")

if __name__ == "__main__":
    make_small_istanbul()
