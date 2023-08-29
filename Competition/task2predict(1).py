import argparse
import json
import time
import pyspark

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
        
    bus_rdd = sc.textFile(train_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], (x['user_id'], x['stars']))) \
        .groupByKey() \
        .mapValues(dict) \
        
    train_dict = train_rdd.collectAsMap()
    bus_dict = bus_rdd.collectAsMap()

    average_user_rating = dict.fromkeys(train_dict.keys())
    average_bus_rating = dict.fromkeys(bus_dict.keys())

    for key in train_dict.keys():
        bus = train_dict[key]
        rate = 0.0
        num_rate = len(bus)
        for rating in bus.values():
            rate+=rating
        average_rate = rate/num_rate
        average_user_rating[key] = average_rate
    
    for key in bus_dict.keys():
        user = bus_dict[key]
        rate = 0.0
        num_rate = len(user)
        for rating in user.values():
            rate+=rating
        average_rate = rate/num_rate
        average_bus_rating[key] = average_rate

    predictions = test_rdd.map(lambda x: predict_item(x, train_dict, model_dict, n_weights)) \
        .filter(lambda x: x is not None) \
        .collect()
        
    with open(output_file, 'w+') as w:
        for i in predictions:
            pred = (average_user_rating[i[0]] + average_bus_rating[i[1]] + i[2])/3
            prediction = {'user_id': i[0], 'business_id': i[1], 'stars': pred}
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
    parser.add_argument('--train_file', type=str, default='./data/train_review_ratings.json')
    parser.add_argument('--test_file', type=str, default='./data/test_review.json')
    parser.add_argument('--model_file', type=str, default='./outputs/model')
    parser.add_argument('--output_file', type=str, default='./outputs/out.json')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.output_file, args.n, sc)
    sc.stop()