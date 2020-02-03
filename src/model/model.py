import torch
import transformers


class DistilBertForQUEST(transformers.DistilBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = transformers.DistilBertModel(config)
        self.pre_classifier = torch.nn.Linear(3097, 3097)
        self.classifier = torch.nn.Linear(3097, config.num_labels)

        self.pre_embedding = torch.nn.Linear(5, 5)
        self.embedding = torch.nn.Linear(5, 5)
        self.dropout = torch.nn.Dropout()
        self.prelu1 = torch.nn.PReLU()
        self.prelu2 = torch.nn.PReLU()
        self.prelu3 = torch.nn.PReLU()
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                category=None,
                sentence_lengths=None,
                newlines=None,
                similarity=None,
                hyperlinks=None,
                first_person=None,
                latex=None,
                brackets=None,
                sentiment=None,
                spell=None,

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
        pre_embed = self.prelu1(pre_embed)
        embedding = self.embedding(pre_embed)

        e = embedding.view(-1, 1, 5)
        sl = sentence_lengths.view(-1, 1, 3)
        n = newlines.view(-1, 1, 2)
        s = similarity.view(-1, 1, 1)
        h = hyperlinks.view(-1, 1, 2)
        fp = first_person.view(-1, 1, 2)
        l = latex.view(-1, 1, 2)
        b = brackets.view(-1, 1, 2)
        se = sentiment.view(-1, 1, 3)
        sp = spell.view(-1, 1, 3)

        cat = torch.cat([h3, h4, h5, h6, e, sl, s, n, h, fp, l, b, se, sp], 2)
        cat = self.prelu2(cat)
        layer = self.pre_classifier(cat)
        layer = self.dropout(layer)
        layer = self.prelu3(layer)
        output = self.classifier(layer)

        return output.reshape((-1, 30))

    def init_weights(self):
        self.distilbert.init_weights()
        self.pre_classifier.reset_parameters()
        self.classifier.reset_parameters()
