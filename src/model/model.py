import torch
import transformers


class RobertaForQUEST(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQUEST, self).__init__(config)
        model_name = 'roberta-base'
        self.bert = transformers.RobertaModel.from_pretrained(model_name,
                                                              config=config)
        self.embedding = torch.nn.Linear(5, 5)
        self.embedding2 = torch.nn.Linear(63, 10)
        self.linear_layer1 = torch.nn.Linear(3163, 2000)
        self.linear_layer2 = torch.nn.Linear(2000, 1000)
        self.linear_layer3 = torch.nn.Linear(1000, 500)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(500, config.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False

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
                token_type_ids=None,
                punctuation=None,
                question_mark=None,
                exclamation=None,
                yes_no=None,
                numeric_answer=None,
                short_keyword=None,
                consequence=None,
                how=None,
                choice=None,
                comparison=None,
                comma=None,
                period=None,
                instruction=None,
                host=None,
                self_reference=None,
                parts_of_speech=None,
                language=None,
                sonic=None,
                reference=None,
                word_part=None,
                ):
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            )

        hidden_states = outputs[2]

        h6 = torch.mean(hidden_states[-1], 1).reshape((-1, 1, 768))
        h5 = torch.mean(hidden_states[-2], 1).reshape((-1, 1, 768))
        h4 = torch.mean(hidden_states[-3], 1).reshape((-1, 1, 768))
        h3 = torch.mean(hidden_states[-4], 1).reshape((-1, 1, 768))

        embedding = self.embedding(category)
        embedding2 = self.embedding2(host)

        e = embedding.view(-1, 1, 5)
        e2 = embedding2.view(-1, 1, 10)
        sl = sentence_lengths.view(-1, 1, 3)
        n = newlines.view(-1, 1, 2)
        s = similarity.view(-1, 1, 1)
        se = sentiment.view(-1,1,3)
        h = hyperlinks.view(-1, 1, 2)
        fp = first_person.view(-1, 1, 2)
        l = latex.view(-1, 1, 2)
        b = brackets.view(-1, 1, 2)
        sp = spell.view(-1, 1, 3)

        p = punctuation.view(-1, 1, 3)
        q = question_mark.view(-1, 1, 3)
        ex = exclamation.view(-1, 1, 3)
        y = yes_no.view(-1, 1, 3)
        na = numeric_answer.view(-1, 1, 3)
        sk = short_keyword.view(-1, 1, 3)
        c = consequence.view(-1, 1, 3)

        hw = how.view(-1, 1, 3)
        ch = choice.view(-1, 1, 3)
        co = comparison.view(-1, 1, 3)
        cm = comma.view(-1, 1, 3)
        pd = period.view(-1, 1, 3)
        i = instruction.view(-1, 1, 3)

        sr = self_reference.view(-1, 1, 2)
        ps = parts_of_speech.view(-1, 1, 3)
        la = language.view(-1, 1, 3)
        sn = sonic.view(-1, 1, 3)
        rf = reference.view(-1, 1, 3)
        wp = word_part.view(-1, 1, 3)

        hcat = torch.cat([h3, h4, h5, h6], 2)

        fcat = torch.cat(
            [e, e2, sl, s, n, h, fp, l, b, sp, p, q, ex, y, na, sk,
             c, hw, ch, co, cm, pd, i, sr, ps, la, sn, rf, wp, se], 2)

        cat = torch.cat([hcat, fcat], 2)
        cat = torch.nn.ReLU()(cat)

        layer1 = self.linear_layer1(cat)
        layer1 = torch.nn.LeakyReLU()(layer1)
        layer2 = self.linear_layer2(layer1)
        layer2 = torch.nn.LeakyReLU()(layer2)
        layer3 = self.linear_layer3(layer2)
        layer3 = torch.nn.LeakyReLU()(layer3)
        logits = self.classifier(layer3)

        return torch.squeeze(logits)
