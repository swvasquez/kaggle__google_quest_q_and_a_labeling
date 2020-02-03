import json

import torch
import yaml
from sklearn.model_selection import KFold
from torch.utils import data


def normalize(values, means, stdevs):
    scaled = []
    for value, mean, stdev in zip(values, means, stdevs):
        scaled.append((value - mean) / stdev)
    return scaled


class QuestDataset(data.Dataset):
    def __init__(self, redis_db, features, labels, targets, indices=None):
        self.redis_db = redis_db
        self.features = features
        self.labels = labels
        self.target = targets
        self.ids = indices if indices is not None else json.loads(redis_db.get(
            'train_ids'))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        qa_id = str(self.ids[index])
        label_item = {}
        feature_item = {}

        for field in self.features['numerical']:
            fload = json.loads(self.redis_db.hget(qa_id, f"train_{field}"))
            means = json.loads(self.redis_db.get(f"train_"
                                                     f"{field}_averages"))
            stdevs = json.loads(
                    self.redis_db.get(f"train_{field}_standard_deviations"))
            fload = normalize(fload, means, stdevs)
            feature_item[field] = torch.tensor(fload)
        for field in self.features['categorical']:
            fload = json.loads(self.redis_db.hget(qa_id, f"train_{field}"))
            feature_item[field] = torch.tensor(fload)
        for field in self.labels:
            lload = json.loads(self.redis_db.hget(qa_id, f"train_{field}"))
            label_item[field] = torch.tensor(lload)

        return feature_item, label_item


class KFoldDataset:

    def __init__(self, config_path, folds, redis_db):
        with config_path.open(mode='r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.redis_cnxn = redis_db
        self.features = config['features']
        self.labels = config['label']
        self.ids = json.loads(redis_db.get('train_ids'))
        self.targets = config['target']

        self.train_datasets = []
        self.test_datasets = []

        kf = KFold(n_splits=folds, shuffle=True)

        for train_indices, test_indices in kf.split(self.ids):
            train_ids = [self.ids[i] for i in train_indices]
            test_ids = [self.ids[i] for i in test_indices]
            self.train_datasets.append(
                QuestDataset(
                    self.redis_cnxn,
                    self.features,
                    self.labels,
                    self.targets,
                    train_ids,
                )
            )

            self.test_datasets.append(
                QuestDataset(
                    self.redis_cnxn,
                    self.features,
                    self.labels,
                    self.targets,
                    test_ids,
                )
            )
