# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:39:09 2023

@author: Mouhamad Ali Elamine
"""

import argparse
import json

parser = argparse.ArgumentParser(description='A1T2')
parser.add_argument('--review_file', type=str, default='./review.json', help='the input file')
parser.add_argument('--business_file', type=str, default='./business.json', help='the input file ')
parser.add_argument('--output_file', type=str, default= './a1t2.json', help='the output file contains your answer')
parser.add_argument('--n', type=int, default=10, help='top n categories with highest average stars')

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
business_file = 'business.json'
r_lines = sc.textFile(review_file).persist()
b_lines = sc.textFile(business_file).persist()
review_rdd = r_lines.map(lambda x: json.loads(x)).persist()
business_rdd = b_lines.map(lambda x: json.loads(x)).persist()

r_rdd = review_rdd.map(lambda x: (x["business_id"],x["stars"]))
b_rdd1 = business_rdd.map(lambda x: (x["business_id"],x["categories"]))
b_rdd2 = b_rdd1.filter(lambda x: x[1] != None)
b_rdd3 = b_rdd2.map(lambda x: [(x[0],categorie) for categorie in x[1].split(', ')])
b_rdd4 = b_rdd3.flatMap(lambda x: x)
join_rdd = r_rdd.join(b_rdd4)
rdd5 = join_rdd.map(lambda x: (x[1][1], x[1][0]))
rdd6 = rdd5.groupByKey().mapValues(lambda x: sum(x)/len(x))
rdd_av = rdd6.sortBy(lambda x: (-x[1],x[0])).take(args.n)

task2 = {
    "result": rdd_av
}

json_task2 = json.dumps(task2, indent=4)

with open(args.output_file, "w") as outfile:
    outfile.write(json_task2)