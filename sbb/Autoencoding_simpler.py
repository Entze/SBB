import random
from collections import namedtuple

import pyro.distributions as dist
import torch
from pyroapi import pyro
from torch import nn
from tqdm import trange

Loader = namedtuple('Loader', ['dataset', 'labels', 'lexicon', 'sentences_len'])


class Decoder(nn.Module):
    def __init__(self, sentence_len, lexicon_len, encoding_dim, hidden_dim):
        super().__init__()
        self.sentence_len = sentence_len
        # setup the two linear transformations used
        self.fc1 = nn.Linear(encoding_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, lexicon_len)
        self.fc22 = nn.Linear(hidden_dim, 1)
        self.fc31 = nn.Linear(sentence_len, 1)

        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z (batch x sentence_length
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Categorical and the output Bernoulli
        # each is of size batch_size x sentence_len x lexicon_len and batch_size x 1 respectively
        sentence_logits = self.fc21(hidden)
        reduction = self.fc22(hidden).squeeze(-1)
        label_prob = self.sigmoid(self.fc31(reduction).squeeze(-1))
        return sentence_logits, label_prob


class Encoder(nn.Module):
    def __init__(self, encoding_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, encoding_dim)
        self.fc22 = nn.Linear(hidden_dim, encoding_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x, y):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.unsqueeze(-1)
        batch_size = x.size(0)
        sentence_length = x.size(1)
        x = torch.cat((y.unsqueeze(-1).unsqueeze(-1).expand(batch_size, sentence_length, 1), x), -1)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = self.fc22(hidden).exp()
        return z_loc, z_scale


class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, sentence_len, lexicon_len, encoding_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(encoding_dim, hidden_dim)
        self.decoder = Decoder(sentence_len, lexicon_len, encoding_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.encoding_dim = encoding_dim

    # define the model p(x|z)p(z)
    def model(self, sentences, labels, sentence_len, obs_flag=True):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", sentences.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros((sentences.shape[0], sentence_len, self.encoding_dim), device=sentences.device)
            z_scale = torch.ones((sentences.shape[0], sentence_len, self.encoding_dim), device=sentences.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(2))

            # decode the latent code z
            sentence_logits, label_prob = self.decoder(z)  # size -> (batch.size, vec_len, lex_len)
            sentences = pyro.sample("obs_sentences", dist.Categorical(logits=sentence_logits).to_event(1),
                                    obs=sentences if obs_flag else None)
            assert labels is None or labels.size(0) == sentences.size(0)
            labels = pyro.sample("obs_labels", dist.Bernoulli(label_prob), obs=labels if obs_flag else None)
            return sentences, labels

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, sentences, labels, sentence_len, obs_flag=True):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", sentences.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(sentences, labels)
            # sample the latent code z
            return pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(2))

    def reconstruct_sentence(self, x, y):
        z_loc, z_scale = self.encoder(x, y)
        z = dist.Normal(z_loc, z_scale).sample()
        sentence_logits, label_prob = self.decoder(z)
        print(sentence_logits)
        print(label_prob)
        words, ind = torch.topk(sentence_logits, 1, dim=-1)
        return ind, torch.round(label_prob)


def train(svi, sentences, labels, sentence_len, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for sentence in sentences:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            sentence = sentence.use_cuda()
        # do ELBO gradient and accumulate loss
    epoch_loss += svi.step(sentences, labels, sentence_len)

    # return epoch loss
    normalizer_train = len(sentences)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, sentences, labels, sentence_len, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for sentence in sentences:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            sentence = sentence.use_cuda()
        # do ELBO gradient and accumulate loss
    test_loss += svi.step(sentences, labels, sentence_len)

    # return epoch loss
    normalizer_test = len(sentences)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def generate_dataset(goods=5, bads=5):
    words = ["bad{}".format(i + 1) for i in range(goods)] + ["good{}".format(i + 1) for i in range(bads)]

    sentences = []
    for i in range(10):
        length = torch.distributions.LogNormal(1., .5).sample().int().item() * 3
        sentences.append(random.choices(words, k=length))

    def label(sentence):
        bs = 0
        for word in sentence:
            if word.startswith('b'):
                bs += 1

        return int(bs > (len(sentence) - bs))

    labels = [label(sentence) for sentence in sentences]

    lexicon = dict()
    sentence_len = max(len(sentence) for sentence in sentences)
    counter = 1
    sentences_translated = []
    for sentence in sentences:
        vec = [0 for _ in range(sentence_len)]
        for i, word in enumerate(sentence):
            if word not in lexicon:
                lexicon[word] = counter
                counter += 1
            vec[i] = lexicon[word]
        sentences_translated.append(vec)
    loader = Loader(torch.tensor(sentences_translated, dtype=torch.float), torch.tensor(labels, dtype=torch.float),
                    lexicon, sentence_len)
    return loader


def main():
    sentences, labels, lexicon, sentence_len = generate_dataset(5, 5)
    lexicon_len = len(lexicon) + 1
    print(sentences, labels, lexicon, sentence_len, lexicon_len)

    vae = VAE(sentence_len, lexicon_len)

    adam = pyro.optim.Adam(dict(lr=1.0e-3))
    elbo = pyro.infer.Trace_ELBO()

    svi = pyro.infer.SVI(vae.model, vae.guide, adam, elbo)

    train_elbo = []
    test_elbo = []
    for epoch in trange(10_000, disable=True):
        total_epoch_loss_train = train(svi, sentences, labels, sentence_len, use_cuda=False)
        train_elbo.append(-total_epoch_loss_train)
        if epoch % 999 == 17:
            print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % 5 == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, sentences, labels, sentence_len, use_cuda=False)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

    print(list(range(sentence_len)), 1,
          vae.reconstruct_sentence(torch.tensor([range(sentence_len)], dtype=torch.float), torch.tensor([1.])))


if __name__ == '__main__':
    main()
