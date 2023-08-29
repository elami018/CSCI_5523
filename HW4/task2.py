import argparse
import time
import pyspark
from itertools import combinations
from collections import defaultdict, deque
import numpy as np

def betweenness(child_nodes, parent_nodes, credit_scores, num_shortest_path):
    edges = []
    for child in child_nodes:
        for parent in parent_nodes[child]:
            credit = credit_scores[child] * (num_shortest_path[parent] / num_shortest_path[child])
            credit_scores[parent] += credit
            edge = tuple(sorted([child, parent]))
            edges.append((edge, credit))
    return edges


def breadth_first_search(root, connected_nodes):
    visited_nodes = [root]
    queue_nodes = []
    parent_nodes = {root: None}
    level = {root: 0}
    num_shortest_path = {root: 1.0}
    credit_score = defaultdict(float)

    for nodes in connected_nodes[root]:
        visited_nodes.append(nodes)
        queue_nodes.append(nodes)
        parent_nodes[nodes] = [root]
        level[nodes] = 1
        num_shortest_path[nodes] = 1
    while queue_nodes:
        node = queue_nodes.pop(0)
        credit_score[node] = 1.0
        num_paths = 0
        for p in parent_nodes[node]:
            num_paths += num_shortest_path[p]
        num_shortest_path[node] = num_paths
        for other_node in connected_nodes[node]:
            if other_node in visited_nodes:
                if level[other_node] == level[node] + 1:
                    parent_nodes[other_node].append(node)
            else:
                visited_nodes.append(other_node)
                queue_nodes.append(other_node)
                parent_nodes[other_node] = [node]
                level[other_node] = level[node] + 1

    return betweenness(visited_nodes[::-1][:-1], parent_nodes, credit_score, num_shortest_path)

def find_communities(nodes, connected_nodes):
    communities = []
    visited_nodes = []
    for each_node in nodes:
        if each_node not in visited_nodes:
            temp = [each_node]
            queue = deque([each_node])
            while queue:
                next_node = queue.popleft()
                for neighbor_node in connected_nodes[next_node]:
                    if neighbor_node not in temp:
                        temp.append(neighbor_node)
                        queue.append(neighbor_node)
                visited_nodes.append(next_node)
            communities.append(temp)
    return communities

def get_modularity(A, communities, nodes, m):
    modularities = []
    for community_nodes in communities:
        community_node_idxs = [nodes.index(node) for node in community_nodes]
        community_A = A[community_node_idxs, :][:, community_node_idxs]
        degrees = np.sum(community_A, axis=1)
        expected_edges = np.outer(degrees, degrees) / (2 * m)
        modularity = np.sum(community_A - expected_edges) / (2 * m)
        modularities.append(modularity)
    total_modularity = np.sum(modularities) / (2 * m)
    return total_modularity

def main(filter_threshold, input_file, output_file1, output_file2, sc):
    rdd = sc.textFile(input_file) \
        .filter(lambda x: not x.startswith("user_id,business_id")) \
        .map(lambda x: (x.split(',')[0], x.split(',')[1])) \
        .persist()

    user_business_dict = rdd.groupByKey() \
        .mapValues(set) \
        .collectAsMap()
    
    users = user_business_dict.keys()
    nodes_edges_dict = defaultdict(list)
    for user1, user2 in combinations(users, 2):
        if len(user_business_dict[user1] & user_business_dict[user2]) >= filter_threshold:
            nodes_edges_dict[user1].append(user2)
            nodes_edges_dict[user2].append(user1)

    
    nodes = sorted(nodes_edges_dict.keys())
    nodes_rdd = sc.parallelize(nodes).persist()
    betweenness_rdd = nodes_rdd \
        .flatMap(lambda x: breadth_first_search(x, nodes_edges_dict)) \
        .groupByKey() \
        .mapValues(sum) \
        .map(lambda x: (x[0], x[1]/2)) \
        .sortBy(lambda x: (-x[1], x[0][0])) \
    
    betweenness = betweenness_rdd.collect()

    with open(output_file1, "w") as output:
        for edge, score in betweenness:
            edge_str = f"('{edge[0]}', '{edge[1]}')"
            output.write(f"{edge_str}, {score}\n")

    n = len(nodes)
    m = sum(len(value) for value in nodes_edges_dict.values())

    A = np.zeros((n,n), dtype=int)
    for i in range(n):
        for connected_node in nodes_edges_dict[nodes[i]]:
            j = nodes.index(connected_node)
            A[i][j] = 1

    best_modularity = -1
    best_communities = []

    visited_nodes = 0
    while (visited_nodes < m):
        communities = find_communities(nodes, nodes_edges_dict)
        modularity_ = get_modularity(A, communities, nodes, m)
        if modularity_ > best_modularity:
            best_modularity = modularity_
            best_communities = communities

        edge = betweenness_rdd.max(lambda x: x[1])
        nodes_edges_dict[edge[0][0]].remove(edge[0][1])
        nodes_edges_dict[edge[0][1]].remove(edge[0][0])
        betweenness_rdd = nodes_rdd \
            .flatMap(lambda x: breadth_first_search(x, nodes_edges_dict))
        if betweenness_rdd.isEmpty():
            break
        betweenness_rdd = betweenness_rdd.groupByKey() \
            .mapValues(sum) \
            .map(lambda x: (x[0], x[1]/2))
        visited_nodes += 1

    
    """ code for saving the output to file in the correct format """
    resultDict = {}
    for community in best_communities:
        community = list(map(lambda userId: "'" + userId + "'", sorted(community)))
        community = ", ".join(community)

        if len(community) not in resultDict:
            resultDict[len(community)] = []
        resultDict[len(community)].append(community)

    results = list(resultDict.items())
    results.sort(key = lambda pair: pair[0])

    output = open(output_file2, "w")

    for result in results:
        resultList = sorted(result[1])
        for community in resultList:
            output.write(community + "\n")
    output.close()


if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--filter_threshold', type=int, default=7, help='')
    parser.add_argument('--input_file', type=str, default='./data/ub_sample_data.csv', help='the input file')
    parser.add_argument('--betweenness_output_file', type=str, default='./out1.txt', help='the output file contains your answers')
    parser.add_argument('--community_output_file', type=str, default='./out2.txt', help='the output file contains your answers')
    args = parser.parse_args()

    main(args.filter_threshold, args.input_file, args.betweenness_output_file, args.community_output_file, sc)
    sc.stop()