import argparse
import json
import time
import pyspark
from numpy import mean, median


def predict_item(test_dict, train_dict, model_dict, n):
    user_id, business_id = test_dict[:2]

    business_rated = train_dict[user_id].keys()

    business_scores = {tuple(sorted((business_id, business))): model_dict[tuple(sorted((business_id, business)))]
                   for business in business_rated
                   if tuple(sorted((business_id, business))) in model_dict}


    sorted_scores = sorted(business_scores.items(), key=lambda x: -x[1])

    neighbors = dict(sorted_scores[:n])

    neighbors_rated = [business for business in business_rated
                          if tuple(sorted((business_id, business))) in neighbors]

    numerator = sum(train_dict[user_id][business] * model_dict[tuple(sorted((business_id, business)))]
                    for business in neighbors_rated)
    denominator = sum(model_dict[tuple(sorted((business_id, business)))]
                      for business in neighbors_rated)

    if not neighbors or denominator == 0:
        prediction = sum(train_dict[user_id].values()) / len(train_dict[user_id])
    else:
        prediction = numerator / denominator

    return user_id, business_id, prediction

def main(train_file, test_file, model_file, output_file, n_weights, sc):

    test_rdd = sc.textFile(test_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id']))

    model_rdd = sc.textFile(model_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: ((x['b1'], x['b2']), x['sim'])) \
        .map(lambda x: (tuple(sorted(x[0])), x[1])) \
        .distinct()
        
    model_dict = model_rdd.collectAsMap()

    train_rdd = sc.textFile(train_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], (x['business_id'], x['stars']))) \
        .groupByKey() \
        .mapValues(dict) \
        
    train_dict = train_rdd.collectAsMap()

    predictions = test_rdd.map(lambda x: predict_item(x, train_dict, model_dict, n_weights)) \
        .filter(lambda x: x is not None)
    
    with open(output_file, 'w+') as w:
        for i in predictions.collect():
            prediction = {'user_id': i[0], 'business_id': i[1], 'stars': i[2]}
            w.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--train_file', type=str, default='./data/train_review.json')
    parser.add_argument('--test_file', type=str, default='./data/val_review.json')
    parser.add_argument('--model_file', type=str, default='./outputs/task2.case1.model')
    parser.add_argument('--output_file', type=str, default='./outputs/task2.case1.val.out')
    parser.add_argument('--time_file', type=str, default='./outputs/time.out')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.output_file, args.n, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))
