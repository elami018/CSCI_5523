# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:39:09 2023

@author: Mouhamad Ali Elamine
"""

import argparse
import json

parser = argparse.ArgumentParser(description='A1T3')
parser.add_argument('--input_file', type=str, default='./review.json', help='the input file')
parser.add_argument('--output_file', type=str, default= './a1t3_customized.json', help='the output file contains your answer')
parser.add_argument('--n_partitions', type=int, default=10, help='the number of partitions')
parser.add_argument('--n', type=int, default=10, help='the threshold of the number of reviews')

args = parser.parse_args()

from pyspark import SparkConf, SparkContext
if __name__ == '__main__':
    sc_conf = SparkConf() \
        .setAppName('task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory','8g') \
        .set('spark.executor.memory','4g')
        
    
    sc = SparkContext.getOrCreate(conf=sc_conf)
    sc.setLogLevel('OFF')

def partitioner(x):
    return hash(x)

review_file = 'review.json'
r_lines = sc.textFile(review_file).persist()
review_rdd = r_lines.map(lambda x: json.loads(x)).persist()
review_rdd1 = review_rdd.map(lambda x: (x["business_id"],1)).persist()

partitioned_rdd = review_rdd1.partitionBy(args.n_partitions, partitioner)
item_num = partitioned_rdd.glom().map(lambda x: len(x)).collect()
partitioned_rdd1 = partitioned_rdd.groupByKey().mapValues(lambda x: sum(x))
partitioned_rdd2 = partitioned_rdd1.filter(lambda x: x[1]>args.n).collect()

task3_customized = {
    "n_partitions":args.n_partitions,
    "n_items":item_num,
    "result": partitioned_rdd2
}

json_task3_customized = json.dumps(task3_customized, indent=4)

with open(args.output_file, "w") as outfile:
    outfile.write(json_task3_customized)