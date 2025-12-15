import numpy as np

def distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def travel_distance(geo):
    return np.sum(distance(geo[:-1, 0], geo[:-1, 1], geo[1:, 0], geo[1:, 1]))


def radius(geo):
    center = np.mean(geo, axis = 0)
    return np.sqrt(np.mean(distance(geo[:, 0], geo[:, 1], center[0], center[1])))


def JSD(P_A, P_B):
    epsilon = 1e-14
    P_A = (P_A / P_A.sum() + epsilon)
    P_B = (P_B / P_B.sum() + epsilon)
    P_merged = 0.5 * (P_A + P_B)
    
    kl_PA_PM = np.sum(P_A * np.log(P_A / P_merged))
    kl_PB_PM = np.sum(P_B * np.log(P_B / P_merged))
    
    jsd = 0.5 * (kl_PA_PM + kl_PB_PM)
    return jsd

def arr_to_distribution(arr, min, max, bins):
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution

def compute_probability_distribution(data):
    unique_elements, counts = np.unique(data, return_counts=True)
    total_counts = np.sum(counts)
    probabilities = counts / total_counts
    return unique_elements, probabilities

def category_jsd(generated_category, real_category):
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)

    p,q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))

    p = np.asarray(p)
    q = np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p:
        p_probs[element] = prob
    
    for element, prob in q:
        q_probs[element] = prob

    jsd_value = JSD(np.array(list(p_probs.values())),np.array(list(q_probs.values())))
    return jsd_value

def grank_jsd(generated_category, real_category,top=1000):
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)
    sorted_indices = np.argsort(-prob_gen)
    gen_category = gen_category[sorted_indices]
    prob_gen = prob_gen[sorted_indices]
    
    sorted_indices = np.argsort(-prob_real)
    real_category = real_category[sorted_indices]
    prob_real = prob_real[sorted_indices]
    
    tt=top
    gen_category=gen_category[:tt]
    prob_gen=prob_gen[:tt]
    real_category=real_category[:tt]
    prob_real=prob_real[:tt]
    p,q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))

    p = np.asarray(p)
    q = np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p:
        p_probs[element] = prob
    
    for element, prob in q:
        q_probs[element] = prob

    jsd_value = JSD(np.array(list(p_probs.values())),np.array(list(q_probs.values())))
    return jsd_value

def evaluation(generated, original):
    generated = np.array(generated)
    original = np.array(original)
    assert len(generated) > 0
    assert len(original) > 0
    max = np.max(generated) if np.max(generated) > np.max(original) else np.max(original)
    p_gen = arr_to_distribution(generated, 0, max, 100)
    p_real = arr_to_distribution(original, 0, max, 100)
    jsd = JSD(p_gen, p_real)
    return jsd

def get_visits(trajs,max_locs):
    visits = np.zeros(shape=(max_locs), dtype=float)
    for t in trajs:
        visits[t] += 1
    visits = visits / np.sum(visits)
    return visits

def get_topk_visits(visits, K):
    locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
    locs_visits.sort(reverse=True, key=lambda d: d[1])
    topk_locs = [locs_visits[i][0] for i in range(K)]
    topk_probs = [locs_visits[i][1] for i in range(K)]
    return np.array(topk_probs), topk_locs


def compute_ovr_vectorized(generated_seqs, real_seqs, po_matrix):
    """使用纯numpy向量化计算Order Violation Rate"""
    # 1. 批量构建位置矩阵
    max_seq_len = max(len(seq['marks']) for seq in generated_seqs)
    batch_size = len(generated_seqs)
    num_categories = po_matrix[0].shape[0]

    # 创建位置矩阵：position_matrix[b, pos, cat] = position if category appears
    position_matrix = np.full((batch_size, num_categories), np.inf)

    for b, seq in enumerate(generated_seqs):
        for pos, cat in enumerate(seq['marks']):
            if cat < num_categories and position_matrix[b, cat] == np.inf:
                position_matrix[b, cat] = pos

                # 2. 批量计算违例
    total_violations = 0
    total_order_edges = 0

    for b, po_matrix in enumerate(po_matrix):
        # 获取所有偏序边的索引
        order_edges = np.where(po_matrix == 1)
        cat_i_list = order_edges[0]
        cat_j_list = order_edges[1]

        # 批量获取位置
        pos_i = position_matrix[b, cat_i_list]
        pos_j = position_matrix[b, cat_j_list]

        # 计算违例：位置有效且违反偏序
        valid_mask = (pos_i != np.inf) & (pos_j != np.inf)
        violation_mask = valid_mask & (pos_i > pos_j)

        total_violations += np.sum(violation_mask)
        total_order_edges += len(cat_i_list)

    return total_violations / total_order_edges if total_order_edges > 0 else 0.0
def Get_Statistical_Metrics(real_data, generated_data, min_seq_len=1,top=1000,po_matrix= None):
    Real_Statistics = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category':[], 'G-RANK':[]}
    Generated_Statistics = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category':[], 'G-RANK':[]}
    JSD = {'Distance': 1.0, 'Radius': 1.0, 'DailyLoc': 1.0, 'Interval': 1.0, 'Category':1.0, 'G-RANK':1.0, 'totalJSD': 6.0}

    # 添加OVR指标
    if po_matrix is not None:
        JSD['OVR'] = 0.0

    data = [generated_data,real_data]
    if len(generated_data)==0 and len(real_data) == 0:
        return JSD
    assert len(generated_data) > 0
    assert len(real_data) > 0

    metrics_dicts = [Generated_Statistics,Real_Statistics]
    for idx,seqs in enumerate(data):
        for seq in seqs:
            if len(seq['gps'])>min_seq_len:
                gps = np.array(seq['gps'])
                metrics_dicts[idx]['Distance'].append(travel_distance(gps))
                metrics_dicts[idx]['Radius'].append(radius(gps))
                metrics_dicts[idx]['DailyLoc'].append(len(set(seq['checkins'])))
                metrics_dicts[idx]['Interval'].extend(np.ediff1d(np.concatenate([[0],seq["arrival_times"]])).tolist())
                metrics_dicts[idx]['Category'].extend(seq['marks'])
                metrics_dicts[idx]['G-RANK'].extend(seq['checkins'])

    for metric in JSD.keys():
        if metric == 'Category':
            JSD[metric] = category_jsd(Generated_Statistics[metric],Real_Statistics[metric])
        elif metric == 'G-RANK':
            JSD[metric] = grank_jsd(Generated_Statistics[metric],Real_Statistics[metric],top)
        elif metric != 'totalJSD':
            JSD[metric] = evaluation(Generated_Statistics[metric],Real_Statistics[metric])
        else:
            break
        JSD['totalJSD'] = sum([JSD[metric] for metric in JSD.keys() if metric != 'totalJSD'])

        if po_matrix is not None:
            JSD['OVR'] = compute_ovr_vectorized(generated_data, real_data, po_matrix)
    return JSD