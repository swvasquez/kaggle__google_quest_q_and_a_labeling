import json

import numpy as np
import scipy
import torch
import transformers
import yaml

from src import project_paths
from src.data import dataset
from src.data.redis_db import redis_cnxn
from src.model import model


def spearman(model, device, dataset, dataloader):
    model.eval()

    with torch.no_grad():
        spearman = []
        predicted = np.zeros((len(dataset), len(dataset.target)))
        target = np.zeros((len(dataset), len(dataset.target)))
        offset = 0

        for data in dataloader:
            features, labels = data
            for field in features:
                features[field] = features[field].to(device)

            for field in labels:
                labels[field] = labels[field].to(device)
            label = labels['target_vector']

            output = model(**features)
            predicted[offset: offset + output.size()[0], :] = \
                output.cpu().numpy()
            target[offset: offset + label.size()[0], :] = \
                label.cpu().numpy()
            offset += + output.size()[0]

            print("Scored:", offset)

        for col in range(len(dataset.target)):
            sprmn = scipy.stats.spearmanr(target[:, col],
                                          predicted[:, col])
            spearman.append(sprmn.correlation)
        avg_spearman = np.average(np.array(spearman))

    return avg_spearman


if __name__ == '__main__':

    # Load configuration file.
    paths = project_paths()
    root_path = paths['root']
    config_path = paths['config']

    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    test_dir = root_path / config['load_model']
    GPU_CAP_TEST = int(config['gpu_capacity_test'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using {device}.")
    test_dir = root_path / config['load_model']
    for filepath in test_dir.iterdir():
        state_dict_path = filepath.resolve().as_posix()

    state_dict = torch.load(state_dict_path)

    model_name = 'distilbert-base-uncased'
    model_config = transformers.DistilBertConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        num_labels=30,
    )

    bnet = model.DistilBertForQUEST.from_pretrained(model_name,
                                                    config=model_config)

    bnet.load_state_dict(state_dict)
    bnet.to(device)

    r = redis_cnxn()
    features = config['features']
    labels = config['label']
    ids = json.loads(r.get('train_ids'))
    targets = config['target']

    test_dataset = dataset.QuestDataset(r, features, labels, targets, False)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=GPU_CAP_TEST,
                                             shuffle=True,
                                             num_workers=0
                                             )

    print(spearman(bnet, device, test_dataset, testloader))
