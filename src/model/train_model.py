import json
import psutil
import subprocess
import time

import numpy as np
import scipy
import transformers
import torch
import yaml

from src import project_paths
from src.data.dataset import QUESTDataset
from src.data.redis_db import redis_cnxn


class DistilBertForQUEST(transformers.DistilBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = transformers.DistilBertModel(config)
        self.pre_classifier = torch.nn.Linear(3077, 3077)
        # self.classifier = torch.nn.Linear(config.dim, config.num_labels)
        # self.dropout = torch.nn.Dropout(config.seq_cassif_dropout)
        self.classifier = torch.nn.Linear(3077, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None,
                inputs_embeds=None, labels=None, features=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask,
            head_mask=head_mask, inputs_embeds=inputs_embeds
        )

        hidden_states = distilbert_output[1]  # (bs, seq_len, dim)
        h6 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h5 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h4 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h3 = hidden_states[-4][:, 0].reshape((-1, 1, 768))

        hcat = torch.cat([h3, h4, h5, h6, features.reshape((-1,1,5))], 2)
        layer = self.pre_classifier(hcat)
        layer = torch.nn.ReLU()(layer)
        output = self.classifier(layer)

        return output.reshape((-1, 30))


class Callback:

    def spearman(self, redis_db, model, device, test_path):
        dataset = QUESTDataset(redis_db)
        batch_size = 30
        trainloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  num_workers=0)

        with torch.no_grad():
            spearman = []
            predicted = np.zeros((len(dataset), len(dataset.target)))
            offset = 0
            for data in trainloader:
                inputs = data[0].to(device)
                features = data[1].to(device)
                labels = data[2].to(device)
                masks = data[3].to(device)
                output = model(inputs, attention_mask=masks, features=features)
                predicted[offset: offset + output.size()[0], :] = \
                    output.cpu().numpy()
                offset += + output.size()[0]
                print(offset)
            for idx, field in enumerate(dataset.target):
                train = np.array(json.loads(redis_db.get(field)))
                sprmn = scipy.stats.spearmanr(train, predicted[:, idx])
                spearman.append(sprmn)
        ssum = sum(v[0] for v in spearman) / 30
        print(ssum, spearman)
        return spearman

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


if __name__ == '__main__':

    # Load configuration file.
    paths = project_paths()
    root_path = paths['root']
    config_path = paths['config']

    with config_path.open(mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_path = root_path / config['save']

    # Get training parameters.
    BATCH_SIZE = int(config['batch_size'])
    GPU_CAP = int(config['gpu_capacity'])
    LEARNING_RATE = float(config['learning_rate'])
    EPOCHS = int(config['epochs'])

    # Use GPU if possible.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using {device}.")

    # Connect to the Redis database that feeds the dataset object.
    r = redis_cnxn()
    dataset = QUESTDataset(r)
    print(GPU_CAP)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=GPU_CAP,
                                              shuffle=True, num_workers=0)
    # Choose model - BERT or DistilBERT
    model_name = 'distilbert-base-uncased'
    model = DistilBertForQUEST

    model_config = transformers.DistilBertConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        num_labels=30,
        sinusoidal_pos_embeds=True
    )

    print(model_config)

    # Initialize model, loss function, and optimizer.
    bnet = model.from_pretrained(model_name, config=model_config)
    bnet.to(device)
    bnet.train()
    print("Model in training mode:", bnet.training)

    torch.cuda.empty_cache()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(bnet.parameters(), lr=0.001, momentum=0.9)
    callback = Callback()
    #callback.spearman(r, bnet, device)
    for epoch in range(EPOCHS):
        # Check temperatures of GPU and CPU
        callback.cool(70, 70, 60)

        minibatches = 0
        running_loss = 0.0
        sample_size = 0
        for i, data in enumerate(trainloader, 0):
            if i % 1000 == 0:
                callback.cool(75, 75, 60)

            # Send inputs to GPU if GPU exists.
            inputs = data[0].to(device)
            features = data[1].to(device)
            labels = data[2].to(device)
            masks = data[3].to(device)

            samples = data[0].shape[0]
            outputs = bnet(inputs, attention_mask=masks, features=features)

            # If the batch size that the GPU can hold is smaller than the
            # desired batch size, accumulate the gradients before updating
            # weights.
            update = BATCH_SIZE // GPU_CAP
            remainder = (len(dataset) % BATCH_SIZE)
            if i > len(dataset) - remainder:
                scale = remainder / samples
            else:
                scale = update

            loss = criterion(outputs, labels) / scale
            loss.backward()
            running_loss += loss.item()
            if i % update == update - 1 or i == len(dataset) - 1:
                optimizer.step()
                minibatches += 1
                # zero the parameter gradients
                optimizer.zero_grad()
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (BATCH_SIZE)))
                running_loss = 0.0
    torch.save(bnet, save_path.resolve().as_posix())
    callback.spearman(r, bnet, device)
    print('Finished Training')
