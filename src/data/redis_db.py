import csv
import itertools
import json
import math
from collections import OrderedDict

import redis
import yaml

import src.features.preprocess as preprocess
from src import project_paths


def redis_cnxn():
    r = redis.Redis(
        host='localhost',
        port=6379
    )
    return r


def grouper(iterable, n, fillvalue=None):
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def preprocess_gen(data_path, config , chunk_size, labels=False):
    input_fields = config['input']
    target_fields = config['target']
    feature_fields = config['features']['categorical'] \
                     + config['features']['categorical']
    label_fields = config['label']
    id_field = config['id'][0]

    with data_path.open(mode='r') as f:
        csv_reader = csv.DictReader(f)
        chunk_iter = grouper(csv_reader, chunk_size)
        samples = 0

        for chunk in chunk_iter:

            ids = []
            exists = 0

            input_dict = OrderedDict()
            target_dict = OrderedDict()
            feature_dict = OrderedDict()
            labels_dict = OrderedDict()

            for field in input_fields:
                input_dict[field] = []

            for field in target_fields:
                target_dict[field] = []

            for row in chunk:
                if row:
                    exists += 1
                    ids.append(row[id_field])
                    for field in input_fields:
                        input_dict[field].append(row[field])
                    if labels:
                        for field in target_fields:
                            target_dict[field].append(row[field])

            print(f"Preprocessing rows {samples} through {samples + exists}.")

            for field in feature_fields:
                feature_function = getattr(preprocess, f"{field}_feature")
                feature_dict[field] = feature_function(input_dict)
            if labels:
                for field in label_fields:
                    label_function = getattr(preprocess, f"{field}_label")
                    labels_dict[field] = label_function(target_dict)
            else:
                labels_dict = None

            samples += exists

            yield ids, feature_dict, labels_dict


def get_categories(train_path):
    categories = set()
    with train_path.open(mode='r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            categories.add(row['category'])
    return categories


def online_mean(sample, iteration, previous_mean):
    next_mean = previous_mean + (sample - previous_mean) / iteration
    return next_mean


def variance_coefficient(previous_coefficient, sample, current_mean,
                         previous_mean):
    next_coefficient = previous_coefficient + (sample - previous_mean) * (
            sample - current_mean)
    return next_coefficient


def standard_dev(coefficient, iteration):
    variance = coefficient / iteration
    return math.sqrt(variance)


def redis_load(redis_db, data_path, config_path, chunk_size, labels):
    data_type = 'train' if labels else 'test'
    print(f"Preprocessing {data_type} data and pushing to Redis.")
    data_gen = preprocess_gen(data_path, config_path, chunk_size, labels)
    redis_db.set(f"{data_type}_ids", json.dumps([]))

    for chunk in data_gen:
        print('Pushing to Redis.\n')
        ids, feature_dict, labels_dict = chunk
        row_ids = json.loads(redis_db.get(f"{data_type}_ids")) + ids
        redis_db.set(f"{data_type}_ids", json.dumps(row_ids))

        for field in feature_dict:
            for idx, row in enumerate(feature_dict[field]):
                redis_db.hset(ids[idx], f"{data_type}_{field}",
                              json.dumps(row))
        if labels:
            for field in labels_dict:
                for idx, row in enumerate(labels_dict[field]):
                    redis_db.hset(ids[idx], f"{data_type}_{field}",
                                  json.dumps(row))

    for field in config['features']['numerical']:
        mean, stdevs = feature_stats(field, redis_db, data_type)
        redis_db.set(f"{data_type}_{field}_averages", json.dumps(mean))
        redis_db.set(f"{data_type}_{field}_standard_deviations",
                     json.dumps(stdevs))


def feature_stats(field, redis_db, data_type):
    stdevs = []
    row_ids = json.loads(redis_db.get(f"{data_type}_ids"))
    for idx, row_id in enumerate(row_ids):
        data = json.loads(redis_db.hget(row_id, f"{data_type}_{field}"))
        if idx == 0:
            means = data
            vars = [0] * len(data)
        else:
            for col in range(len(means)):
                prev_mean = means[col]
                means[col] = online_mean(data[col], idx, means[col])
                vars[col] = variance_coefficient(vars[col], data[col],
                                                 means[col], prev_mean)
    for idx1, _ in enumerate(vars):
        stdevs.append(standard_dev(vars[idx1], len(row_ids)))
    return means, stdevs


if __name__ == '__main__':
    paths = project_paths()
    root_path = paths['root']
    config_path = paths['config']

    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_path = root_path / config['train']
    test_path = root_path / config['test']

    r = redis_cnxn()
    redis_load(r, train_path, config, 512, True)
    redis_load(r, test_path, config, 512, False)
