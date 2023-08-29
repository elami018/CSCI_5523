# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:39:09 2023

@author: Mouhamad Ali Elamine
"""

import argparse, json, time 
from itertools import combinations, chain
from collections import Counter
from pyspark import SparkConf, SparkContext

def generate_candidates_of_size_n_in_chunk(candidates_tuples):
    def compare_prefix(candidates_tuples):
        new_candidates = []
        for i, prefix_tuple in enumerate(candidates_tuples[:-1]):
            for other_tuple in candidates_tuples[i+1:]:
                if prefix_tuple[:-1] == other_tuple[:-1]:
                    new_candidate = tuple(sorted(set(prefix_tuple) | set(other_tuple)))
                    new_candidates.append(new_candidate)
        return new_candidates
    if len(candidates_tuples) == 0:
        return []
    if len(candidates_tuples[0]) == 1:
        flattened_list = [elem for tup in candidates_tuples for elem in tup]
        new_candidates = list(combinations(flattened_list, 2))
    else:
        new_candidates = compare_prefix(candidates_tuples)
    return new_candidates

def get_itemset_count(chunk, candidates):
    itemset_count = {}
    for basket in chunk:
        for candidate in candidates:
            if set(candidate).issubset(basket):
                itemset_count[candidate] = itemset_count.get(candidate, 0) + 1
    return itemset_count

def get_itemset_count_for_chunks(chunk, candidates,frequent_singleton):
    itemset_count = {}
    for basket in chunk:
        basket = list(set(basket).intersection({x[0] for x in frequent_singleton}))
        for candidate in candidates:
            if set(candidate).issubset(basket):
                itemset_count[candidate] = itemset_count.get(candidate, 0) + 1
    return itemset_count

def get_frequent_itemsets_of_all_sizes_in_chunk(chunk):
    def get_frequent_singletons_in_chunk(chunk, chunk_support):
        candidate_singleton = Counter()
        frequent_singleton = []
        for basket in chunk:
            candidate_singleton += Counter(basket)
        for item in candidate_singleton:
            if candidate_singleton[item] >= chunk_support:
                frequent_singleton.append(item)
        return frequent_singleton
    
    def get_frequent_itemsets_of_size_n_in_chunk(chunk, chunk_support, frequent_singleton):
        candidates_list = frequent_singleton
        frequent_itemsets_by_size = {}
        n = 1
        while len(candidates_list):
            itemset_count = get_itemset_count_for_chunks(chunk, candidates_list, frequent_singleton)
            frequent_itemsets = {item: count for item, count in itemset_count.items() if count >= chunk_support}
            frequent_itemsets_by_size[n] = list(frequent_itemsets)
            n += 1
            candidates_list = generate_candidates_of_size_n_in_chunk(sorted(frequent_itemsets))
        return list(frequent_itemsets_by_size.values())

    chunk_list = list(chunk)
    chunk_support = args.s * len(chunk_list) / num_baskets
    frequent_singleton = get_frequent_singletons_in_chunk(chunk_list, chunk_support)
    frequent_singleton_tuple = [(element,) for element in frequent_singleton]
    frequent_itemsets = get_frequent_itemsets_of_size_n_in_chunk(chunk_list, chunk_support, frequent_singleton_tuple)
    return frequent_itemsets


def lexicographic_ordering(lst):
    
    def group_by_length(lst):
        length_dict = {}
        for elem in lst:
            length = len(elem)
            if length not in length_dict:
                length_dict[length] = [elem]
            else:
                length_dict[length].append(elem)
        return list(length_dict.values())

    def string_sorting(lst):
        for sublist in lst:
            sublist.sort()
        return lst
    
    return string_sorting(group_by_length(lst))



if __name__ == '__main__':
    sc_conf = SparkConf() \
        .setAppName('task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory','8g') \
        .set('spark.executor.memory','4g')
    sc = SparkContext.getOrCreate(conf=sc_conf)
    sc.setLogLevel('OFF')
    parser = argparse.ArgumentParser(description='A1T1')
    parser.add_argument('--c', type=int, default= 1, help='Case 1/2')
    parser.add_argument('--s', type=int, default= 8, help='support')
    parser.add_argument('--input_file', type=str, default='small2.csv',help='the input file')
    parser.add_argument('--output_file', type=str, default='t1s4', help='the output file contains answer')

    args = parser.parse_args()

    start_time = time.time()

    small_file = args.input_file
    r_lines = sc.textFile(small_file)
    small_rdd = r_lines.filter(lambda x: not x.startswith("user_id,business_id"))

    if args.c == 1:
        small_rdd1 = small_rdd.map(lambda x: (x.split(',')[0], x.split(',')[1])).groupByKey().mapValues(list)
        total_baskets_rdd = small_rdd1.map(lambda x: x[1]).persist()

    elif args.c == 2:
        small_rdd1 = small_rdd.map(lambda x: (x.split(',')[1], x.split(',')[0])).groupByKey().mapValues(list)
        total_baskets_rdd = small_rdd1.map(lambda x: x[1]).persist()
    
    num_baskets = total_baskets_rdd.count()

    frequent_itemsets_in_chunks = total_baskets_rdd.mapPartitions(get_frequent_itemsets_of_all_sizes_in_chunk)
    total_candidates = frequent_itemsets_in_chunks.flatMap(lambda x: x).map(lambda x: (x,1)).groupByKey().map(lambda x: x[0]).collect()
    total_frequents = total_baskets_rdd.mapPartitions(lambda x: (get_itemset_count(x, total_candidates)).items())
    real_candidates = total_frequents.map(lambda x: x).groupByKey().mapValues(lambda x: sum(x))
    display_candidates = real_candidates.map(lambda x: x[0]).collect()
    real_frequents = real_candidates.filter(lambda x: x[1]>=args.s).map(lambda x: x[0]).collect()

    candidates_grouped = lexicographic_ordering(display_candidates)
    frequents_grouped = lexicographic_ordering(real_frequents)

    end_time = time.time()

    task1 = {
        "Candidates": candidates_grouped,
        "Frequent Itemsets": frequents_grouped,
        "Runtime": end_time - start_time
    }
    json_task1 = json.dumps(task1, indent=4)

    with open(args.output_file, "w") as outfile:
        outfile.write(json_task1)