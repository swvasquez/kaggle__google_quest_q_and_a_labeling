import json

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
        self.pre_classifier = torch.nn.Linear(config.dim, config.dim)
        self.classifier = torch.nn.Linear(config.dim, config.num_labels)
        self.dropout = torch.nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None,
                inputs_embeds=None, labels=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask,
            head_mask=head_mask, inputs_embeds=inputs_embeds
        )

        hidden_states = distilbert_output[1]  # (bs, seq_len, dim)
        h6 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h5 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h4 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h3 = hidden_states[-4][:, 0].reshape((-1, 1, 768))
        hcat = torch.cat([h3, h4, h5, h6], 1)
        pool = torch.mean(hcat, 1)

        pooled = self.dropout(pool)
        output = self.classifier(pooled)

        return output


class BertForQUEST(transformers.BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = transformers.BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size,
                                          self.config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = bert_outputs[2]

        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h9 = hidden_states[-4][:, 0].reshape((-1, 1, 768))
        hcat = torch.cat([h9, h10, h11, h12], 1)
        pool = torch.mean(hcat, 1)

        pooled = self.dropout(pool)
        output = self.classifier(pooled)

        return output


def callback(redis_db, model, device):
    dataset = QUESTDataset(redis_db)
    batch_size = 30
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0)

    with torch.no_grad():
        spearman = []
        predicted = np.zeros((len(dataset), len(dataset.target)))
        offset = 0
        for batch in trainloader:
            output = model(batch[0].to(device), attention_mask=batch[2].to(
                device))
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
        num_labels=30
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

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        minibatches = 0
        running_loss = 0.0
        sample_size = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)
            masks = data[2].to(device)

            samples = data[0].shape[0]
            # forward + backward + optimize

            outputs = bnet(inputs, attention_mask=masks)

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
    callback(r, bnet, device)
    print('Finished Training')
