import argparse, pyspark, json, time
from itertools import combinations
from numpy import random

def generateHash(a, b, m, p=10099):
    def hash(x):
        return (a * x + b) % p % m
    return hash

def minHashing(x, hash_functions):
    '''
    Takes in a tuple containing a business ID and a list of user IDs, and applies
    the minhash algorithm using the given hash functions to generate a signature
    for the business.
    
    :param x: A tuple containing a business ID and a list of user IDs
    :param hash_functions: A list of hash functions to use for minhashing
    :return: A tuple containing the business ID and its minhash signature
    '''
    
    # Unpack the input tuple to extract the business ID and list of user IDs
    business_id, user_ids = x[0], x[1]
    
    # For each hash function in the hash functions list, apply it to the list of user IDs
    # and find the minimum hash value for that function
    min_hash_values = [min(map(h, user_ids)) for h in hash_functions]
    
    # Return a tuple containing the business ID and its minhash signature
    return business_id, min_hash_values

def slice_bands(x, n_rows, n_bands):
    """
    Slices the MinHash signature into bands and returns tuples 
    of band IDs and document IDs.
    
    :param x: A tuple containing a document ID and its MinHash signature
    :param n_rows: The number of rows per band
    :param n_bands: The number of bands to create
    :return: A list of tuples, where each tuple contains a band ID and a document ID
    """
    # Extract the MinHash signature from the input tuple
    signature = x[1]
    
    # Slice the signature into bands of n_rows rows each
    bands = [(tuple(signature[i:i+n_rows])) for i in range(0, n_bands)]
    
    # Create tuples of band IDs and document IDs for each band in the signature
    band_doc_tuples = [((i, b), x[0]) for i, b in enumerate(bands)]
    
    # Return the list of band_doc_tuples
    return band_doc_tuples


def main(input_file, output_file, jac_thr, n_bands, n_rows, sc):
    n_hash = n_bands*n_rows

    rdd = sc.textFile(input_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id']))

    user_ID_dict = rdd.map(lambda x: x[0]) \
        .distinct() \
        .zipWithUniqueId() \
        .collectAsMap()

    business_ID_rdd = rdd.map(lambda x: x[1]) \
        .distinct() \
        .zipWithUniqueId()
    
    business_ID_dict = business_ID_rdd.collectAsMap()

    ID_business_dict = business_ID_rdd.map(lambda x: (x[1], x[0])) \
        .collectAsMap()

    business_user_rdd = rdd.map(lambda x: (business_ID_dict[x[1]], user_ID_dict[x[0]])) \
        .groupByKey()
    business_user_dict = business_user_rdd.collectAsMap()

    hash_functions = []
    a = random.randint(1, 1000000, size=n_hash)
    b = random.randint(1, 1000000, size=n_hash)
    for i in range(n_hash):
        hash_functions.append(generateHash(a[i], b[i], len(user_ID_dict)))

    minHashed_business_user_rdd = business_user_rdd.map(lambda x: minHashing(x, hash_functions))

    sliced_rdd = minHashed_business_user_rdd.flatMap(lambda x: slice_bands(x, n_rows, n_bands))

    candidates = sliced_rdd.groupByKey() \
        .map(lambda x: list(x[1])) \
        .flatMap(lambda x: list(combinations(x, 2))) \
        .map(lambda x: tuple(sorted(x))) \
        .distinct()

    similar_pairs = candidates.flatMap(lambda x: [(x[0], x[1], len(set(business_user_dict[x[0]]) \
        .intersection(business_user_dict[x[1]])) / len(set(business_user_dict[x[0]]) \
        .union(business_user_dict[x[1]]))) if len(set(business_user_dict[x[0]]) \
        .union(business_user_dict[x[1]])) > 0 else None]) \
        .filter(lambda x: x[2] >= jac_thr)

    with open(output_file, 'w') as w:
        for item in similar_pairs.collect():
            similar_pair = {'b1': ID_business_dict[item[0]], 'b2': ID_business_dict[item[1]], 'sim': item[2]}
            w.write(json.dumps(similar_pair) + '\n')


if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--input_file', type=str, default='./train_review.json')
    parser.add_argument('--output_file', type=str, default='./task1.out')
    parser.add_argument('--time_file', type=str, default='./task1.time')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--n_bands', type=int, default=50)
    parser.add_argument('--n_rows', type=int, default=2)
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.threshold, args.n_bands, args.n_rows, sc)
    sc.stop()

    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))
