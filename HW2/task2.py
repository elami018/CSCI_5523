# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:39:09 2023

@author: Mouhamad Ali Elamine
"""

import argparse, json, time
from itertools import combinations
from pyspark import SparkConf, SparkContext


def generate_candidates_of_size_n_in_chunk(candidates, n):
    new_candidates = set()
    if n == 1:
        for basket in candidates:
            for item in basket:
                new_candidates.add(frozenset([item]))
    else:
        for itemset1 in candidates:
            for itemset2 in candidates:
                candidate_of_size_n = itemset1 | itemset2
                if len(candidate_of_size_n) == n:
                    new_candidates.add(frozenset(candidate_of_size_n))
    return new_candidates

def generate_candidate_pairs(chunk, frequent_singleton):
    candidate_pairs = set()
    singletons = {elem for s in frequent_singleton for elem in s}
    for basket in chunk:
        comb_pair = list(combinations(basket, 2))
        for pair in comb_pair:
            if singletons.issuperset(pair):
                candidate_pairs.add(frozenset(pair))
    return candidate_pairs

def remove_superset_if_subset_not_frequent(candidates, prev_frequent):
    frequent_items = set()
    for item in candidates:
        subsets = {item - {elem} for elem in item}
        if all([subset in prev_frequent for subset in subsets]):
            frequent_items.add(item)
    return frequent_items

def get_itemset_count(chunk, candidates):
    itemset_count = {}
    if not candidates:
        return itemset_count
    else:
        for basket in chunk:
            for candidate in candidates:
                if candidate.issubset(basket):
                    itemset_count[candidate] = itemset_count.get(candidate, 0) + 1
    return itemset_count

def get_frequent_itemsets(itemset_count, support):
    filtered_dict = {k: v for k, v in itemset_count.items() if v >= support}
    return set(filtered_dict.keys())

def get_frequent_itemsets_of_all_sizes(chunk):
    chunk = list(chunk)
    chunk_support = len(chunk)*args.s/num_baskets
    frequent_itemsets = {}
    singletons = generate_candidates_of_size_n_in_chunk(chunk, 1)
    singletons_count = get_itemset_count(chunk, singletons)
    frequent_singletons = get_frequent_itemsets(singletons_count, chunk_support)
    frequent_itemsets_n_size = frequent_singletons
    n = 1
    while len(frequent_itemsets_n_size):
        frequent_itemsets[n] = frequent_itemsets_n_size
        print(f"generating candidates of size {n+1}")
        if n == 1:
            potential_candidates_n_size = generate_candidate_pairs(chunk, frequent_singletons)
        else:
            potential_candidates_n_size = generate_candidates_of_size_n_in_chunk(frequent_itemsets_n_size, n+1)
        print(f"removing superset of size {n+1}")
        true_candidates_n_size = remove_superset_if_subset_not_frequent(potential_candidates_n_size, frequent_itemsets_n_size)
        print(f"getting itemset count")
        itemset_count = get_itemset_count(chunk, true_candidates_n_size)
        frequent_itemsets_n_size = get_frequent_itemsets(itemset_count, chunk_support)
        n+=1
    
    frequent_set = set()
    for _, items in frequent_itemsets.items():
        for item in items:
            frequent_set.add(item)
    return frequent_set

def lexicographic_ordering(lst):
    def group_by_length(lst):
        length_dict = {}
        for elem in lst:
            length = len(elem)
            length_dict.setdefault(length, []).append(list(elem))
        return list(length_dict.values())
    def string_sorting(lst):
        for sublist in lst:
            sublist.sort()
        return lst
    return string_sorting(group_by_length(lst))

if __name__ == '__main__':
    sc_conf = SparkConf() \
        .setAppName('task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory','8g') \
        .set('spark.executor.memory','4g')
    sc = SparkContext.getOrCreate(conf=sc_conf)
    sc.setLogLevel('OFF')

    parser = argparse.ArgumentParser(description='Task2')
    parser.add_argument('--k', type=int, default= 10,help=' filter out qualified users')
    parser.add_argument('--s', type=int, default=10, help='support')
    parser.add_argument('--input_file', type=str, default='user_business.csv', help='the input file ')
    parser.add_argument('--output_file', type=str, default= './t2k10s10_test.json', help='the output file contains answer')

    args = parser.parse_args()

    start = time.time()
    data_file = args.input_file
    data_lines = sc.textFile(data_file)
    data_rdd = data_lines.filter(lambda x: not x.startswith("user_id,business_id"))
    data_rdd1 = data_rdd.map(lambda x: (x.split(',')[0], x.split(',')[1]))
    total_baskets_rdd = data_rdd1.groupByKey().mapValues(set).filter(lambda x: len(x[1])>args.k).map(lambda x: x[1])
    num_baskets = total_baskets_rdd.count()
    frequent_itemsets_in_chunks = total_baskets_rdd.mapPartitions(get_frequent_itemsets_of_all_sizes).distinct()
    candidates = frequent_itemsets_in_chunks.collect()       
    total_frequents = total_baskets_rdd.mapPartitions(lambda x: get_itemset_count(x, candidates).items()).groupByKey().mapValues(lambda x: sum(x)).filter(lambda x: x[1]>=args.s).map(lambda x: x[0]).collect()
    candidates_grouped = lexicographic_ordering(candidates)
    frequents_grouped = lexicographic_ordering(total_frequents)
    
    end = time.time()
    task2 = {
        "Candidates": candidates_grouped,
        "Frequent Itemsets": frequents_grouped,
        "Runtime": end - start
    }
    json_task2 = json.dumps(task2, indent=4)
    with open(args.output_file, "w") as outfile:
        outfile.write(json_task2)
