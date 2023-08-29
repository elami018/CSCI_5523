import argparse
import json
import time
import pyspark
from itertools import combinations
from numpy import mean
from numpy.linalg import norm
import numpy as np


def pearsonCorrelation(business_pair, business_user_rating, users):
    user_rating1 = business_user_rating[business_pair[0]]
    user_rating2 = business_user_rating[business_pair[1]]

    user_intersect = users[business_pair[0]] & users[business_pair[1]]

    rating_intersect1 = np.array([user_rating1[i] for i in user_intersect])
    rating_intersect2 = np.array([user_rating2[i] for i in user_intersect])

    rating_average1 = mean(rating_intersect1)
    rating_average2 = mean(rating_intersect2)
    norm1 = norm(rating_intersect1 - rating_average1)
    if norm1 != 0:
        norm2 =norm(rating_intersect2 - rating_average2)
        if norm2 !=0:
            denominator = norm1 * norm2
            numerator = sum((rating_intersect1 - rating_average1) * (rating_intersect2 - rating_average2))
            weights = numerator / denominator
            return (business_pair[0], business_pair[1], weights) if weights > 0 else None
        return None
    return None

def main(train_file, model_file, co_rated_thr, sc):    
    rdd = sc.textFile(train_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], x['user_id'], x['stars'])) \
        .persist()
    
    business_user_rating = rdd.map(lambda x: (x[0], (x[1], x[2]))) \
            .groupByKey() \
            .mapValues(dict) \
            .collectAsMap()
    
    business_users = rdd.map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(set) \
        .persist()
    
    businesses = business_users.keys().collect()
    business_pairs = []
    business_users_dict = business_users.collectAsMap()
    for business1, business2 in combinations(businesses, 2):
        if len(business_users_dict[business1] & business_users_dict[business2]) >= co_rated_thr:
            business_pairs.append((business1, business2))

    business_pairs = sc.parallelize(business_pairs) \
        .persist()
    
    weights = business_pairs.map(lambda x: pearsonCorrelation(x, business_user_rating, business_users_dict)) \
        .filter(lambda x: x is not None) \
        .collect()

    with open(model_file, 'w+') as w:
        for item in weights:
            w.write(json.dumps({'b1': item[0], 'b2': item[1], 'sim': item[2]}) + '\n')


if __name__ == '__main__':
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--train_file', type=str, default='./data/train_review_ratings.json')
    parser.add_argument('--model_file', type=str, default='./outputs/model')
    parser.add_argument('--m', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.model_file, args.m, sc)
    sc.stop()
