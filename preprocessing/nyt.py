import os
from os.path import join
import json


def get_small_test_file():
    file_path = "/home/cleeag/relation_extraction/data/open_nre_nyt/test.json"
    output_path = "/home/cleeag/relation_extraction/data/open_nre_nyt/small_test.json"
    dataset = json.load(open(file_path, 'r'))
    small_dataset = []
    for i, mention in enumerate(dataset):
        if i == 100: break
        small_dataset.append(mention)

    json.dump(small_dataset, fp=open(output_path, 'w'))

    small_dataset = json.load(open(output_path, 'r'))

    for mention in small_dataset:
        print(mention)



if __name__ == '__main__':
    get_small_test_file()