import argparse
import time
import pyspark
from graphframes import *
from itertools import combinations
from pyspark.sql import SparkSession

def main(filter_threshold, input_file, output_file, sc):
    SparkSession(sc)
    rdd = sc.textFile(input_file) \
        .filter(lambda x: not x.startswith("user_id,business_id")) \
        .map(lambda x: (x.split(',')[0], x.split(',')[1])) \

    user_business_dict = rdd.groupByKey() \
        .mapValues(set) \
        .collectAsMap()
    
    users = user_business_dict.keys()
    nodes = set()
    edges = []
    for user1, user2 in combinations(users, 2):
        if len(user_business_dict[user1] & user_business_dict[user2]) >= filter_threshold:
            nodes.add(user1)
            nodes.add(user2)
            edges.append((user1, user2))
            edges.append((user2, user1))

    nodes = sc.parallelize(list(nodes)) \
        .map(lambda x: (x,)) \
        .toDF(['id'])
    
    edges = sc.parallelize(edges).toDF(['src','dst'])

    graph = GraphFrame(nodes, edges)
    communities = graph.labelPropagation(maxIter=5) \
        .rdd.map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .mapValues(list) \
        .map(lambda x: x[1]) \
        .collect()

    for i in communities:
        print(i)

    """ code for saving the output to file in the correct format """
    resultDict = {}
    for community in communities:
        community = list(map(lambda userId: "'" + userId + "'", sorted(community)))
        community = ", ".join(community)

        if len(community) not in resultDict:
            resultDict[len(community)] = []
        resultDict[len(community)].append(community)

    results = list(resultDict.items())
    results.sort(key = lambda pair: pair[0])

    output = open(output_file, "w")

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
    parser.add_argument('--community_output_file', type=str, default='./result.txt', help='the output file contains your answers')
    args = parser.parse_args()

    main(args.filter_threshold, args.input_file, args.community_output_file, sc)
    sc.stop()