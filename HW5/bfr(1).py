import random, os, argparse, json, csv
from sklearn.cluster import KMeans
import numpy as np
from time import time
from itertools import combinations
# random.seed(100)

def load_file(data_path, filenames):
    global counter
    file_path = os.path.join(data_path, filenames[counter])
    with open(file_path, 'r') as f:
        file_contents = f.readlines()
    dp = list(map(lambda x: x.strip("\n").split(','), file_contents))
    dp = [(point[0],(list(map(lambda x:float(x), point[1:])))) for point in dp]
    counter+=1
    return dp

def kMeans_(dp, n_cluster, multiplier=1):
    data = np.array([item[1] for item in dp])
    n = min(len(dp), n_cluster*multiplier)
    kmeans = KMeans(n_clusters=n, max_iter=500, init='k-means++').fit(data)
    clusters_labels = kmeans.labels_
    clusters_centroids = kmeans.cluster_centers_
    clusters = [[] for _ in range(n)]
    for i, cluster_label in enumerate(clusters_labels):
        clusters[cluster_label].append(dp[i])
    clusters_stds = []
    for cluster in clusters:
        cluster_arr = np.array([item[1] for item in cluster])
        clusters_stds.append(np.std(cluster_arr, axis=0))
    clusters_stds = np.array(clusters_stds)
    return clusters, clusters_centroids, clusters_stds 

def Mahalanobis_distance(point, centroids, std_devs):
    normalized_distance = (point - np.array(centroids))/np.array(std_devs)
    MD = np.sqrt(np.sum(normalized_distance**2, axis=1))
    return MD

def generate_stats(Set):
    summary = []
    means = []
    stds = []
    total_sum = 0

    for cluster in Set:
        data = [x[1] for x in cluster]
        N = len(data)
        total_sum += N
        SUM = np.sum(data, axis=0)
        cluster_squared = np.power(data,2)
        SUMSQ = np.sum(cluster_squared, axis=0)
        means.append(SUM/N)
        stds.append(np.sqrt(SUMSQ/N - (SUM/N)**2))
        summary.append(list(x[0] for x in cluster))
    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)
    return summary, total_sum, means, stds

def first_round(dp, n_cluster, alpha):
    # sample_size = int(len(dp) * 0.2)
    # indices = list(range(len(dp)))
    # random.shuffle(indices)
    # sample_indices = indices[:sample_size]
    # remaining_indices = indices[sample_size:]
    # sample = [dp[i] for i in sample_indices]
    # remaining = [dp[i] for i in remaining_indices]
    # dp = remaining

    # clusters, clusters_centroids, clusters_stds = kMeans_(sample, n_cluster)

    sample_size = int(len(dp) * 0.2)
    indices = list(range(len(dp)))
    random.shuffle(indices)
    sample_indices = indices[:sample_size]
    remaining_indices = indices[sample_size:]
    sample = [dp[i] for i in sample_indices]
    # remaining = [dp[i] for i in remaining_indices]
    # dp = remaining

    clusters, clusters_centroids, clusters_stds = kMeans_(sample, n_cluster)

    while any(len(cluster) < 10 for cluster in clusters):
        sample_size = int(len(dp) * 0.2)
        indices = list(range(len(dp)))
        random.shuffle(indices)
        sample_indices = indices[:sample_size]
        remaining_indices = indices[sample_size:]
        sample = [dp[i] for i in sample_indices]

        clusters, clusters_centroids, clusters_stds = kMeans_(sample, n_cluster)
        for cluster in clusters:
            print(len(cluster))

    remaining = [dp[i] for i in remaining_indices]
    dp = remaining
    d = len(sample[0][1])
    threshold = alpha * np.sqrt(d)
    DS = [[] for _ in range(n_cluster)]
    for i, cluster in enumerate(clusters):
        DS[i] = cluster

    # print(f"Len of DP Initial: {len(dp)}")
    for key, point in enumerate(dp):
        MD = Mahalanobis_distance(np.array(point[1]), clusters_centroids, clusters_stds)
        closest_centroid = np.argmin(MD)
        closest_centroid_distance = MD[closest_centroid]      
        if closest_centroid_distance < threshold:
            DS[closest_centroid].append(point)
            del dp[key]
    # print(f"Len of DP after CS Assignment: {len(dp)}")
    return DS, dp

def main(input_path, n_cluster, out_file1, out_file2, alpha=2):
    header = ['round_id', 'nof_cluster_discard',
              'nof_point_discard', 'nof_cluster_compression',
               'nof_point_compression', 'nof_point_retained']

    with open(out_file2, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    num_files = len(os.listdir(input_path))
    filenames = sorted(os.listdir(input_path))
    first_dp = load_file(input_path, filenames)
    DS, first_dp_rem = first_round(first_dp, n_cluster, alpha)
    d = len(first_dp_rem[0][1])
    threshold = alpha * np.sqrt(d)
    DS_summary, DS_sum, DS_means, DS_stds = generate_stats(DS)

    CS = []
    RS = []
    clusters, _, _ = kMeans_(first_dp_rem, n_cluster, 10)
    for cluster in clusters:
        if len(cluster) == 1:
            RS.append(cluster[0])
        else:
            CS.append(cluster)
    CS_summary, CS_sum, CS_means, CS_stds = generate_stats(CS)

    # print("FIRST FILE DONE")
    for _ in range(num_files-1):
        intermediate = [counter, len(DS), DS_sum, len(CS), CS_sum, len(RS)]

        with open(out_file2, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(intermediate)

        dp = load_file(input_path, filenames)
        # print(f"Len of DP Initial: {len(dp)}")
        for key, point in enumerate(dp):
            MD = Mahalanobis_distance(np.array(point[1]), DS_means, DS_stds)
            closest_centroid_idx = np.argmin(MD)
            closest_centroid_distance = MD[closest_centroid_idx] 
            if closest_centroid_distance < threshold:
                DS[closest_centroid_idx].append(point)
                del dp[key]


        # print(f"Len of DP after DS Assignment: {len(dp)}")
        DS_summary, DS_sum, DS_means, DS_stds = generate_stats(DS)

        for key, point in enumerate(dp):
            MD = Mahalanobis_distance(np.array(point[1]), CS_means, CS_stds)
            closest_centroid_idx = np.argmin(MD)
            closest_centroid_distance = MD[closest_centroid_idx]  
            if closest_centroid_distance < threshold: 
                CS[closest_centroid_idx].append(point)
                del dp[key]

        # print(f"Len of DP after CS Assignment: {len(dp)}")
        CS_summary, CS_sum, CS_means, CS_stds = generate_stats(CS)
        RS += dp
        if len(RS) != 0:
            clusters, _, _ = kMeans_(RS, n_cluster, 10)
            RS = []
            for cluster in clusters:
                if len(cluster) == 1:
                    RS.append(cluster[0])
                else:
                    CS.append(cluster)

            CS_summary, CS_sum, CS_means, CS_stds = generate_stats(CS)

            CS_pairs_idx = combinations(range(len(CS)), 2)

            merged = []
            merged_idx = set()
            for idx1, idx2 in CS_pairs_idx:
                if (idx1 not in merged_idx) and (idx2 not in merged_idx):
                    normalized_distance = (CS_means[idx1] - np.array(CS_means[idx2]))/np.array(CS_stds[idx1])
                    MD = np.sqrt(np.sum(normalized_distance**2, axis=0))
                    if MD < threshold:
                        merged.append((idx1, idx2))
                        merged_idx.add(idx1)
                        merged_idx.add(idx2)

            for pair in merged:
                idx1, idx2 = pair
                cluster_merge = CS[idx1] + CS[idx2]
                CS.append(cluster_merge)

            for idx in sorted(merged_idx, reverse=True):
                del CS[idx]
            
            CS_summary, CS_sum, CS_means, CS_stds = generate_stats(CS)
        
        # for cluster in DS:
        #     print(len(cluster))
            
    # print("LAST FILE DONE")
    remove_idx = set()
    for key, cluster_mean in enumerate(CS_means):
        MD = Mahalanobis_distance(np.array(cluster_mean), DS_means, DS_stds)
        closest_centroid_idx = np.argmin(MD)
        closest_centroid_distance = MD[closest_centroid_idx]
        if closest_centroid_distance < threshold:
            merged = CS[key] + DS[closest_centroid_idx]
            DS[closest_centroid_idx] = merged
            remove_idx.add(key)
    
    for i in sorted(remove_idx, reverse=True):
        del CS[i]

    DS_summary, DS_sum, DS_means, DS_stds = generate_stats(DS)    
    CS_summary, CS_sum, CS_means, CS_stds = generate_stats(CS)    
    intermediate = [counter, len(DS), DS_sum, len(CS), CS_sum, len(RS)]

    DS_mapping = {point: i for i, s in enumerate(DS_summary) for point in s}
    CS_mapping = {point: -1 for i, s in enumerate(CS_summary) for point in s}
    RS_mapping = {point[0]: -1 for point in RS}
    result = {}
    result.update(DS_mapping)
    result.update(CS_mapping)
    result.update(RS_mapping)

    with open(out_file1, 'w') as f:
        json.dump(result, f)
        f.write("\n")

    with open(out_file2, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(intermediate)

if __name__ == '__main__':
    start = time()
    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_path', type=str, default='./data/test2', help='the folder containing the files of data points')
    parser.add_argument('--n_cluster', type=int, default=10, help='the number of clusters')
    parser.add_argument('--out_file1', type=str, default='./out1_t2.json', help='the output file of cluster results')
    parser.add_argument('--out_file2', type=str, default='./out2_t2.csv', help='the output file of intermediate results')
    args = parser.parse_args()
    counter = 0
    main(args.input_path, args.n_cluster, args.out_file1, args.out_file2, alpha=3)
    end = time()
    print('Runtime: ', end-start)