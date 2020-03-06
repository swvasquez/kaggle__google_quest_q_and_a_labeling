import datetime
import subprocess
import time

import numpy as np
import psutil
import scipy
import torch
import transformers
import yaml

from src import project_paths
from src.data.dataset import KFoldDataset
from src.data.redis_db import redis_cnxn
from src.model import model


class Callback:

    def spearman(self, model, device, dataset, dataloader):
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
            avg_spearman = np.average(np.nan_to_num(np.array(spearman)))

            display = ''
            for idx, field in enumerate(dataset.target):
                left_just = 44 if spearman[idx] < 0 else 45
                col = f"{field}".ljust(left_just) + f"{spearman[idx]}\n"
                display += col
            print(spearman)
            print(f"\naverage: {avg_spearman}")
            print(display)

        return avg_spearman

    def _gpu_temp(self):
        command = ['nvidia-smi', '--query-gpu=temperature.gpu',
                   '--format=csv,noheader']
        temp = subprocess.run(command, check=True,
                              stdout=subprocess.PIPE).stdout
        temp = int(temp.decode('utf-8').strip('\n'))
        return temp

    def _core_temps(self):
        temps = [int(core[1]) for core in psutil.sensors_temperatures()[
            'coretemp']]

        return temps

    def cool(self, gpu_max, cpu_max, wait_time):
        while self._gpu_temp() > gpu_max or any((core > cpu_max for core in
                                                 self._core_temps())):
            print("Waiting for system to cool down.")
            print(f"GPU temp: {self._gpu_temp()}")
            print(f"CPU temps: "
                  f"{','.join((str(t) for t in self._core_temps()))}")
            time.sleep(wait_time)

    def save(self, model, score, dir):
        now = datetime.datetime.now()
        timestamp = now.strftime('%d-%b-%Y_%H_%M')
        rounded_score = str(round(score, 3)).replace('.', '-')
        filename = f"{timestamp}_saved_model_{rounded_score}"
        ouput_path = dir / filename

        torch.save(model.state_dict(), ouput_path.resolve().as_posix())


if __name__ == '__main__':

    # Load configuration file.
    paths = project_paths()
    root_path = paths['root']
    config_path = paths['config']

    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_dir = root_path / config['save']

    # Get training parameters.
    BATCH_SIZE = int(config['batch_size'])
    GPU_CAP_TRAIN = int(config['gpu_capacity_train'])
    GPU_CAP_TEST = int(config['gpu_capacity_test'])
    LEARNING_RATE = float(config['learning_rate'])
    EPOCHS = int(config['epochs'])
    FOLDS = int(config['folds'])

    # Use GPU if possible.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using {device}.")

    # Connect to the Redis database that feeds the dataset object.
    r = redis_cnxn()

    # Create train/test datasets for K-fold cross-validation.
    kfold_dataset = KFoldDataset(config_path, FOLDS, r)
    train_folds = kfold_dataset.train_datasets
    test_folds = kfold_dataset.test_datasets

    model_name = 'roberta-base'
    model_config = transformers.RobertaConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        num_labels=30,
    )
    print(model_config)
    bnet = model.RobertaForQUEST(config=model_config)

    # Load saved model if desired.
    test_dir = root_path / config['load_model']
    state_dict_path = None
    for filepath in test_dir.iterdir():
        state_dict_path = filepath.resolve().as_posix()
    if state_dict_path:
        print('Loading saved model.')
        state_dict = torch.load(state_dict_path)
        bnet.load_state_dict(state_dict)

    bnet.to(device)
    bnet.train()

    # Establish cost function and optimizer. Make sure to move model to
    # GPU before defining optimizer.
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(bnet.parameters(), lr=LEARNING_RATE)

    callback = Callback()
    fold_spearman = []

    for fold in range(FOLDS):
        # For each fold, reset model parameters and update the dataloaders
        # with new datasets.
        if fold > 0:
            break
            bnet.init_weights()

        bnet.train()

        train_dataset = train_folds[fold]
        test_dataset = test_folds[fold]

        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=GPU_CAP_TRAIN,
                                                  shuffle=True,
                                                  num_workers=0
                                                  )
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=GPU_CAP_TEST,
                                                 shuffle=True,
                                                 num_workers=0
                                                 )
        # Calculate initial score on fold.
        print(f"Calculating untrained score on fold {fold}.")
        callback.spearman(bnet, device, test_dataset, testloader)

        for epoch in range(EPOCHS):
            # Check temperatures of GPU and CPU
            callback.cool(70, 70, 60)

            # Shuffle dataset between epochs.
            train_dataset.shuffle()
            minibatches = 0
            running_loss = 0.0

            # if epoch % 10 == 9:
            #     LEARNING_RATE = LEARNING_RATE / 10
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = LEARNING_RATE
            #
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])

            for i, data in enumerate(trainloader, 0):
                # Cool intermitantly.
                if i % 1000 == 0:
                    callback.cool(75, 75, 60)

                # Send inputs to GPU if GPU exists.
                features, labels = data

                for field in features:
                    features[field] = features[field].to(device)

                for field in labels:
                    labels[field] = labels[field].to(device)
                label = labels['target_vector']

                # Push data to model.
                outputs = bnet(**features)

                # If the batch size that the GPU can hold is smaller than the
                # desired batch size, accumulate the gradients before updating
                # weights.
                update = BATCH_SIZE // GPU_CAP_TRAIN
                remainder = (len(train_dataset) % BATCH_SIZE)
                if i > len(train_dataset) - remainder:
                    rescale = remainder / samples
                else:
                    rescale = update
                loss = criterion(outputs, label) / rescale
                loss.backward()
                running_loss += loss.item()

                if i % update == update - 1 or i == len(train_dataset) - 1:
                    optimizer.step()

                    # Zero the parameter gradients.
                    optimizer.zero_grad()
                    print('[Fold %d, Epoch %d, Minibatches %5d] BCE loss: '
                          '%.3f' %
                          (fold + 1, epoch + 1, minibatches + 1,
                           running_loss / (BATCH_SIZE)))
                    running_loss = 0.0
                    minibatches += 1

            spearman = callback.spearman(bnet, device, test_dataset,
                                         testloader)
            callback.save(bnet, spearman, save_dir)
        fold_spearman.append(spearman)
        callback.save(bnet, spearman, save_dir)

    print(fold_spearman)
    print('Finished Training')
