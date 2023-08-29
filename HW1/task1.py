# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:39:09 2023

@author: Mouhamad Ali Elamine
"""

import argparse
import json
import re

parser = argparse.ArgumentParser(description='A1T1')
parser.add_argument('--input_file', type=str, default='./review.json', help='the input file ')
parser.add_argument('--output_file', type=str, default= './a1t1.json', help='the output file contains your answer')
parser.add_argument('--stopwords', type=str, default='./stopwords',help='the file contains the stopwords')
parser.add_argument('--y', type=int, default=2018, help='year')
parser.add_argument('--m', type=int, default=10, help='top m users')
parser.add_argument('--n', type=int, default=10, help='top n frequent words')

args = parser.parse_args()
stopwords = args.stopwords
f = open(stopwords)
#initialize stopwords list with empty string to account for trailing spaces
stopwords_list = ['']
#append stop words to list
for line in f:
    stopwords_list.append(line.rstrip())
f.close()

#create context
from pyspark import SparkConf, SparkContext
if __name__ == '__main__':
    sc_conf = SparkConf() \
        .setAppName('task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory','8g') \
        .set('spark.executor.memory','4g')
    sc = SparkContext.getOrCreate(conf=sc_conf)
    sc.setLogLevel('OFF')

input_file = 'review.json'
#persist lines -> read input file only once
lines = sc.textFile(input_file).persist()
#persist review_rdd -> convert json into dictionary only once
review_rdd = lines.map(lambda x: json.loads(x)).persist()

#part A
rdd_count = review_rdd.count()

#part B
year_rdd = review_rdd.filter(lambda x: x["date"][:4] == str(args.y)).persist()
year_count = year_rdd.count()


#part C
user_rdd = review_rdd.map(lambda x: x["user_id"]).distinct()
user_count = user_rdd.count()

#part D
topm_rdd = review_rdd.map(lambda x: (x["user_id"],1) )
topm_rdd1 = topm_rdd.groupByKey().mapValues(lambda x: sum(x))
topm_rdd2 = topm_rdd1.sortBy(lambda x: (-x[1], x[0])).take(args.m)

#part E
text_rdd = review_rdd.map(lambda x: re.sub('[\(\[,.!?:;\]\)]','', x["text"]))
text_rdd1 = text_rdd.flatMap(lambda x: x.lower().split())
text_rdd2 = text_rdd1.filter(lambda x: x not in stopwords_list)
text_rdd3 = text_rdd2.map(lambda x: (x,1))
text_rdd4 = text_rdd3.groupByKey().mapValues(lambda x: sum(x))
text_rdd5 = text_rdd4.sortBy(lambda x: (-x[1],x[0]))
text_rdd6 = text_rdd5.map(lambda x: x[0]).take(args.n)

task1 = {
    "A": rdd_count,
    "B":year_count,
    "C":user_count,
    "D":topm_rdd2,
    "E":text_rdd6
}

json_task1 = json.dumps(task1, indent=4)

with open(args.output_file, "w") as outfile:
    outfile.write(json_task1)