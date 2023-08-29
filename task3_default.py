# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:39:09 2023

@author: Mouhamad Ali Elamine
"""

import argparse
import json

parser = argparse.ArgumentParser(description='A1T3')
parser.add_argument('--input_file', type=str, default='./review.json', help='the input file')
parser.add_argument('--output_file', type=str, default= './a1t3_default.json', help='the output file contains your answer')
parser.add_argument('--n', type=int, default=5, help='the threshold of the number of reviews')

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


review_file = 'review.json'
r_lines = sc.textFile(review_file).persist()
review_rdd = r_lines.map(lambda x: json.loads(x)).persist()

partition_num = review_rdd.getNumPartitions()
item_num = review_rdd.glom().map(lambda x: len(x)).collect()
rdd = review_rdd.map(lambda x: (x["business_id"],1))
rdd1 = rdd.groupByKey().mapValues(lambda x: sum(x))
rdd2 = rdd1.filter(lambda x: x[1]>args.n).collect()

task3_default = {
    "n_partitions":partition_num,
    "n_items":item_num,
    "result": rdd2
}

json_task3_default = json.dumps(task3_default, indent=4)

with open(args.output_file, "w") as outfile:
    outfile.write(json_task3_default)