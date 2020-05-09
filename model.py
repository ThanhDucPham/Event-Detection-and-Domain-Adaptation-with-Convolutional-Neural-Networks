import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from utils import load_vocab


class Config(object):
    def __init__(self):
        self.num_epoch = 5
        self.learning_rate = 0.01
        self.weight_decay = 1e-4
        self.adam_eps = 1e-8
        self.batch_size = 50
        self.eval_batch_size = 128
        self.nstep_logging = 500
        self.warmup_steps = 2000
        self.change_lr_steps = 100
        self.max_restart = 4
        self.seed = 150

        self.window_size = 15
        self.max_sent = self.window_size * 2 + 1
        self.kernel_sizes = [2, 3, 4, 5]
        self.nfeature_maps = 150
        self.dropout_rate = 0.5
        self.entity_dim = 50
        self.position_dim=50
        self.max_l2norm = 3
        self.norm_type = 2
        self.dropout = 0.5
        self.num_hidden_layers = 2
        self.use_highway = True
        self.vocab_word_size = 14078

        self.fine_tune=True
        self.EPAD_ID = 0
        self.WPAD_ID = 0
        self.LAB_PAD_ID = -100
        self.EPAD = 'PAD'
        self.WPAD = 'PAD'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dir_train = 'data/train.json'
        self.dir_dev = 'data/dev.json'
        self.test_dir = 'data/test.json'
        self.dir_word2vec = 'data/trimmed_word2vec_new.txt'
        self.dir_data = 'data/'
        self.output_dir = 'results/cnn_2015/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.load_data()
        try:
            print('Currently working on ', torch.cuda.get_device_name(0))
        except:
            pass


    def load_data(self):
        vocab_event = load_vocab(self.dir_data + 'vocab_event.txt', hasPad=False)
        self.vocab_event = dict({'O': 0})
        for key in vocab_event:
            if key[2:] not in self.vocab_event and key != 'O':
                self.vocab_event.update({key[2:]: len(self.vocab_event)})
        # 34 classes includes event type + None type
        self.vocab_ner = load_vocab(self.dir_data + 'vocab_ner_tail.txt')
        self.num_class_events = len(self.vocab_event)
        self.num_class_entities = len(self.vocab_ner)


    def set_seed(self, seed=None):
        if seed is None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


#@title High Way net

class HighWay(nn.Module):
    def __init__(self, dim, use_highway=True, dropout=0.5):
        super(HighWay, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_highway = use_highway
        self.trans = nn.Linear(dim, dim)
        if use_highway:
            self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
        f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
        and ⨀ is element-wise multiplication
        """

        h = torch.tanh(self.trans(x))
        if self.use_highway:
            g = torch.sigmoid(self.gate(x))
            x = g * h + (1 - g) * x
        else:
            x = h
        x = self.dropout(x)

        return x


class CNNModel(nn.Module):
    def __init__(self, config, class_weights=None, pretrained_embeddings=None):
        self.config = config
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_word_size, 300, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(pretrained_embeddings)
            self.word_embeddings.weight.requires_grad = config.fine_tune

        self.ner_embeddings = nn.Embedding(config.num_class_entities, config.entity_dim, padding_idx=0,
                                           max_norm=config.max_l2norm, norm_type=config.norm_type)

        self.position_embeddings = nn.Embedding(config.window_size + 2, config.position_dim, padding_idx=0,
                                                max_norm=config.max_l2norm,
                                                norm_type=config.norm_type)  # vocab_position: pad_idx + window_size + center

        embedding_dim = 300 + config.entity_dim + config.position_dim
        self.cnn = nn.ModuleList()
        for kernel_size in config.kernel_sizes:
            self.cnn.append(nn.Conv2d(in_channels=1, out_channels=config.nfeature_maps,
                                      kernel_size=(kernel_size, embedding_dim), padding=(kernel_size - 1, 0)))

        self.dropout = nn.ModuleList()
        for _ in range(4):
            self.dropout.append(nn.Dropout(config.dropout))

        # self.hidden_layers = nn.ModuleList()
        # for _ in range(config.num_hidden_layers):
        #     self.hidden_layers.append(HighWay(dim=config.nfeature_maps * len(config.kernel_sizes),
        #                                       use_highway=config.use_highway,
        #                                       dropout=config.dropout))

        self.classifier = nn.Linear(in_features=config.nfeature_maps * len(config.kernel_sizes) + 300,
                                    out_features=config.num_class_events)
        self.loss_func = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self,
                input_ids,
                input_ners,
                input_positions,
                labels=None
                ):

        word_embeddings = self.word_embeddings(input_ids)
        ner_embeddings = self.ner_embeddings(input_ners)
        position_embeddings = self.position_embeddings(input_positions)
        embeddings = torch.cat((word_embeddings, ner_embeddings, position_embeddings), dim=-1)
        embeddings = embeddings.unsqueeze(1)  # batch_size, 1, len_sent, hid_dim

        total_cnn_outs = []
        for conv in self.cnn:
            cnn_out = conv(embeddings)  # batch, nfeature_maps, new_dim, 1
            cnn_out = torch.squeeze(cnn_out, dim=-1)
            cnn_out = F.max_pool1d(cnn_out, cnn_out.size(2))
            total_cnn_outs.append(cnn_out)

        total_cnn_outs = torch.cat(total_cnn_outs, dim=2)
        cnn_out = total_cnn_outs.view(total_cnn_outs.size(0), -1)
        cnn_out = self.dropout[3](cnn_out)
        cnn_out = torch.cat((cnn_out, word_embeddings[:, self.config.window_size]), dim=-1)
        logits = self.classifier(cnn_out)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_func(logits, labels)
            outputs += (loss,)

        return outputs  # (logits, loss)

    def params_requires_grad(self):

        return [params for params in self.parameters() if params.requires_grad == True]

