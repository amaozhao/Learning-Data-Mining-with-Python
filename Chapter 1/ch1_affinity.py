# coding: utf-8

import numpy as np
from collections import defaultdict
from operator import itemgetter

dataset_filename = "affinity_dataset.txt"

X = np.loadtxt(dataset_filename)
n_samples, n_features = X.shape

features = ["bread", "milk", "cheese", "apples", "bananas"]

num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:  # 此人购买了苹果
        num_apple_purchases += 1
print("{0} people bought Apples".format(num_apple_purchases))

rule_valid = 0
rule_invalid = 0
for sample in X:
    if sample[3] == 1:  # This person bought Apples
        if sample[4] == 1:
            # This person bought both Apples and Bananas
            rule_valid += 1
        else:
            # This person bought Apples, but not Bananas
            rule_invalid += 1
print("{0} cases of the rule being valid were discovered".format(rule_valid))
print("{0} cases of the rule being invalid were discovered".format(
    rule_invalid))

# The Support is the number of times the rule is discovered.
support = rule_valid
confidence = float(rule_valid) / num_apple_purchases
print("The support is {0} and the confidence is {1:.3f}.".format(
    support, confidence))
# Confidence can be thought of as a percentage using the following:
print("As a percentage, that is {0:.1f}%.".format(100 * confidence))

valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0:
            continue
        # Record that the premise was bought in another transaction
        num_occurances[premise] += 1
        for conclusion in range(n_features):
            # It makes little sense to measure if X -> X.
            if premise == conclusion:
                continue
            if sample[conclusion] == 1:
                # This person also bought the conclusion item
                valid_rules[(premise, conclusion)] += 1
            else:
                # This person bought the premise, but not the conclusion
                invalid_rules[(premise, conclusion)] += 1
support = valid_rules
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    confidence[(premise, conclusion)] = valid_rules[
        (premise, conclusion)] / float(num_occurances[premise])


def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(
        premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")


# 排序
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)


for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)
