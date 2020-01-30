import torch
import transformers


class DistilBertForQUEST(transformers.DistilBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = transformers.DistilBertModel(config)
        self.pre_classifier = torch.nn.Linear(3082, 3082)
        self.classifier = torch.nn.Linear(3082, config.num_labels)

        self.pre_embedding = torch.nn.Linear(5, 5)
        self.embedding = torch.nn.Linear(5, 5)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                category=None,
                sentence_lengths=None,
                newlines=None
                ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds

        )

        hidden_states = distilbert_output[1]
        h6 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h5 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h4 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h3 = hidden_states[-4][:, 0].reshape((-1, 1, 768))

        pre_embed = self.pre_embedding(category)
        pre_embed = torch.nn.ReLU()(pre_embed)
        embedding = self.embedding(pre_embed)
        embedding = torch.nn.ReLU()(embedding)

        e = embedding.view(-1, 1, 5)
        s = sentence_lengths.view(-1, 1, 3)
        n = newlines.view(-1, 1, 2)

        cat = torch.cat([h3, h4, h5, h6, e, s, n], 2)

        layer = self.pre_classifier(cat)
        layer = torch.nn.ReLU()(layer)
        output = self.classifier(layer)

        return output.reshape((-1, 30))

    def init_weights(self):
        self.distilbert.init_weights()
        self.pre_classifier.reset_parameters()
        self.classifier.reset_parameters()
