import argparse
import functools
import math
import random
import string
import time
from collections import namedtuple, defaultdict
from copy import deepcopy

import pandas as pd
import preprocessor
import pyro.distributions as dist
import torch
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from pyroapi import pyro
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

Loader = namedtuple('Loader', ['dataset', 'labels'])


class Decoder(nn.Module):
    def __init__(self, sentence_len, lexicon_len, encoding_dim, hidden_dim):
        super().__init__()
        self.sentence_len = sentence_len
        # setup the two linear transformations used
        self.fc1 = nn.Linear(encoding_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, lexicon_len)
        self.fc22 = nn.Linear(hidden_dim, 1)
        self.fc31 = nn.Linear(sentence_len, 1)
        torch.nn.init.normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc1.bias)
        torch.nn.init.normal_(self.fc21.weight)
        torch.nn.init.normal_(self.fc21.bias)
        torch.nn.init.normal_(self.fc22.weight)
        torch.nn.init.normal_(self.fc22.bias)
        torch.nn.init.normal_(self.fc31.weight)
        torch.nn.init.normal_(self.fc31.bias)

        # setup the non-linearities
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, z):
        # define the forward computation on the latent z (batch x sentence_length
        # first compute the hidden units
        hidden = self.tanh(self.fc1(z))
        # return the parameter for the output Categorical and the output Bernoulli
        # each is of size batch_size x sentence_len x lexicon_len and batch_size x 1 respectively
        sentence_logits = self.fc21(hidden)
        reduction = self.tanh(self.fc22(hidden).squeeze(-1))
        label_prob = self.sigmoid(self.fc31(reduction).squeeze(-1))
        return sentence_logits, label_prob


class Encoder(nn.Module):
    def __init__(self, encoding_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, encoding_dim)
        self.fc22 = nn.Linear(hidden_dim, encoding_dim)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc21.weight)
        nn.init.normal_(self.fc21.bias)
        nn.init.normal_(self.fc22.weight)
        nn.init.normal_(self.fc22.bias)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.unsqueeze(-1)
        batch_size = x.size(0)
        sentence_length = x.size(1)
        x = torch.cat((y.unsqueeze(-1).unsqueeze(-1).expand(batch_size, sentence_length, 1), x), -1)
        # then compute the hidden units
        hidden = self.tanh(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.tanh(self.fc22(hidden)).exp()
        assert not torch.any(z_loc.isnan())
        assert not torch.any(z_scale.isnan())
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
        words, ind = torch.topk(sentence_logits, 1, dim=-1)
        return ind, torch.round(label_prob)

    def predict(self, x):
        while x.dim() < 2:
            x = x.unsqueeze(0)
        z_loc_zero, z_scale_zero = self.encoder(x, torch.zeros(x.size(0)))
        z_loc_one, z_scale_one = self.encoder(x, torch.ones(x.size(0)))
        z_zero = dist.Normal(z_loc_zero, z_scale_zero).sample()
        z_one = dist.Normal(z_loc_one, z_scale_one).sample()
        _, label_prob_zero = self.decoder(z_zero)
        _, label_prob_one = self.decoder(z_one)
        zero_prob = (label_prob_zero + (1 - label_prob_one)) / 2
        one_prob = (label_prob_one + (1 - label_prob_zero)) / 2
        if zero_prob > one_prob:
            return 0
        elif one_prob > zero_prob:
            return 1
        else:
            return random.getrandbits(1)


def learn(svi, sentences, labels, sentence_len, batch_size=16, mode='TRAIN', use_cuda=False, pbar=None):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for sentence in sentences:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            sentence = sentence.cuda()
        # do ELBO gradient and accumulate loss
    dataset_size = sentences.size(0)
    processed = 0
    iterations = 0
    while processed < dataset_size:
        batch_end = min(dataset_size, processed + batch_size)
        batch_sentences = sentences[processed:batch_end]
        batch_labels = labels[processed:batch_end]
        if mode == 'TRAIN':
            epoch_loss += svi.step(batch_sentences, batch_labels, sentence_len)
        else:
            epoch_loss += svi.evaluate_loss(batch_sentences, batch_labels, sentence_len)
        if pbar is not None:
            pbar.update()
        processed = batch_end
        iterations += 1

    # return epoch loss
    normalizer = len(sentences)
    total_epoch_loss = epoch_loss / normalizer
    return total_epoch_loss


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


def preprocess_pipeline(sentence):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(preprocessor.clean(sentence))
    ts = []
    for token in tokens:
        ts.extend(token.split('-'))
    tokens = deepcopy(ts)
    ts = []
    for token in tokens:
        ts.extend(token.split('/'))
    tokens = ts
    return [stemmer.stem(lemmatizer.lemmatize(token.lower())) for token in tokens if
            token.lower() not in (list(string.punctuation) + list(string.digits) + ["url", "amp", "should'vetaken",
                                                                                    '', u'', "''", "..", "...",
                                                                                    "....",
                                                                                    ".....",
                                                                                    "......", ",,", "'s", "`s",
                                                                                    "'nt", "n't",
                                                                                    "'m", "'re", "\\5", "``", ' '])]


def std_ordinal_encoder(count, length):
    return count


def build_lexicon(sentences, labels, length=None, encoder=std_ordinal_encoder):
    if length is None:
        length = len(sentences)
    assert len(sentences) == len(labels) == length
    lexicon = defaultdict(lambda: defaultdict(int))
    counter = 1
    for i in range(length):
        label = labels[i]
        sentence = sentences[i]
        for word in sentence:
            if word not in lexicon:
                lexicon[word]['encoding'] = encoder(counter, length)
                counter += 1
            lexicon[word][label] += 1
    return lexicon


def encode(sentence, lexicon, sentence_len=None, default=None):
    if sentence_len is None:
        sentence_len = len(lexicon)
    return [lexicon[word].get('encoding', None) for word in sentence] + [default for _ in
                                                                         range(sentence_len - len(sentence))]


def swap_elems(array1, array2, pos1=0, pos2=0):
    array1.iloc[pos1], array2.iloc[pos2] = array2.iloc[pos2], array1.iloc[pos1]


def balance_datasets(x1, x2, y1, y2, bias=None):
    if bias is None:
        bias0 = 0
        bias1 = 0
        for y in y1:
            if y == 0:
                bias0 += 1
            else:
                bias1 += 1
    else:
        bias0, bias1 = bias

    searching_for = 0 if bias0 > bias1 else 1
    idx1 = 0
    abort = False


    with tqdm(total=abs(bias0 - bias1), desc="Balancing dataset") as pbar:
        while abs(bias0 - bias1) >= 2 and not abort:
            while y1.iloc[idx1] != searching_for:
                idx1 += 1
                if len(y1) <= idx1:
                    abort = True
                    break
            if abort:
                break
            bias0 -= 1 - searching_for
            bias1 -= searching_for
            x1.drop(index=x1.iloc[idx1], inplace=True)
            y1.drop(index=x2.iloc[idx1], inplace=True)
            pbar.update()

    return x1, x2, y1, y2


def prepare_dataset(path, sep=',', rename_columns=None, replace=None, data_column='sentence', label_column='label',
                    train_size=None, test_size=None, evaluate_size=None, random_state=None):
    starttime = time.monotonic()
    learn_size = train_size + test_size if train_size is not None and test_size is not None else None
    print("Reading dataset", end='', flush=True)
    starttime = time.monotonic()
    data = pd.read_csv(path, sep=sep, engine='python').rename(columns=rename_columns).replace(replace)
    print(". Done in {:.2f}s".format(time.monotonic() - starttime))
    print("Preprocessing dataset", end='', flush=True)
    starttime = time.monotonic()
    data[data_column] = data[data_column].apply(preprocess_pipeline)
    print(". Done in {:.2f}s".format(time.monotonic() - starttime))

    data_X, data_y = data[data_column], data[label_column]

    print("Building lexicon", end='')
    starttime = time.monotonic()
    lexicon = build_lexicon(data_X, data_y)
    print(". Done in {:.2f}s".format(time.monotonic() - starttime))
    sentence_len = max(len(sentence) for sentence in data[data_column])
    print("Encode datapoints", end='')
    starttime = time.monotonic()
    data[data_column] = data[data_column].apply(
        functools.partial(encode, lexicon=lexicon, sentence_len=sentence_len, default=0))
    print(". Done in {:.2f}s".format(time.monotonic() - starttime))

    X_learn, X_evaluate, y_learn, y_evaluate = train_test_split(data_X, data_y, train_size=learn_size,
                                                                test_size=evaluate_size, random_state=random_state)

    #bias0 = (y_learn == 0).sum()
    #bias1 = (y_learn == 1).sum()
    #X_learn, X_evaluate, y_learn, y_evaluate = balance_datasets(X_learn, X_evaluate, y_learn, y_evaluate,
    #                                                            (bias0, bias1))
    #bias0 = (y_learn == 0).sum()
    #bias1 = (y_learn == 1).sum()
    #print("Remaining bias {} zeros vs. {} ones".format(bias0, bias1))
    X_train, X_test, y_train, y_test = train_test_split(X_learn, y_learn, train_size=train_size, test_size=test_size,
                                                        random_state=random_state)

    train_loader = Loader(torch.tensor(list(X_train.to_numpy()), dtype=torch.float),
                          torch.tensor(list(y_train.to_numpy()), dtype=torch.float))
    test_loader = Loader(torch.tensor(list(X_test.to_numpy()), dtype=torch.float),
                         torch.tensor(list(y_test.to_numpy()), dtype=torch.float))
    evaluate_loader = Loader(torch.tensor(list(X_evaluate.to_numpy()), dtype=torch.float),
                             torch.tensor(list(y_evaluate.to_numpy()), dtype=torch.float))
    return sentence_len, lexicon, train_loader, test_loader, evaluate_loader


def main(path, sep=None, epochs=1_000, test_frequency=10, minibatch_size=16, loss_threshold=None, use_cuda=False):
    print(
        f"Training on dataset '{path}' for {epochs} epochs with minibatches of size {minibatch_size}. Cuda={use_cuda}")
    sentence_len, lexicon, train_loader, test_loader, evaluate_loader = prepare_dataset(path, sep=sep, rename_columns={
        "subtask_a": "label", "tweet": "sentence"}, replace={'OFF': 1, 'NOT': 0})
    lexicon_len = len(lexicon) + 1
    train_len = train_loader.dataset.size(0)
    test_len = test_loader.dataset.size(0)
    evaluation_len = evaluate_loader.dataset.size(0)
    total_len = train_len + test_len + evaluation_len

    vae = VAE(sentence_len, lexicon_len)

    adam = pyro.optim.Adam(dict(lr=1.0e-3))
    elbo = pyro.infer.Trace_ELBO()

    svi = pyro.infer.SVI(vae.model, vae.guide, adam, elbo)

    train_elbo = []
    test_elbo = []

    minibatches_train = math.ceil(train_len / minibatch_size)
    minibatches_test = math.ceil(test_len / minibatch_size)
    train_steps = epochs * minibatches_train
    test_steps = math.ceil(epochs / test_frequency) * minibatches_test
    total_steps = train_steps + test_steps
    epoch_digits = max(1, round(math.log10(epochs)))

    print(
        "Optimizing on {} ({:.2f}%) datapoints."
        " Testing on {} ({:.2f}%) datapoints. Evaluating on {} ({:.2f}%) datapoints.".format(
            train_len, (100. * train_len / total_len),
            test_len, (100. * test_len / total_len),
            evaluation_len, (100. * evaluation_len / total_len)
        ))

    with tqdm(total=total_steps, desc="  Training") as pbar:
        epoch = 0
        last_loss = (loss_threshold or 1) + 1
        while epoch < epochs and (loss_threshold is None or last_loss > loss_threshold):
            total_epoch_loss_train = learn(svi, train_loader.dataset, train_loader.labels, sentence_len,
                                           batch_size=minibatch_size,
                                           use_cuda=use_cuda, pbar=pbar)
            train_elbo.append(-total_epoch_loss_train)

            pbar.write(("[Loss train {:" + str(epoch_digits) + "}]: {}").format(epoch, total_epoch_loss_train))
            if epoch % test_frequency == 0:
                # report test diagnostics
                total_epoch_loss_test = learn(svi, test_loader.dataset, test_loader.labels, sentence_len,
                                              batch_size=minibatch_size, mode='TEST',
                                              use_cuda=use_cuda, pbar=pbar)
                last_loss = total_epoch_loss_test
                test_elbo.append(-total_epoch_loss_test)
                pbar.write(("[Loss test  {:" + str(epoch_digits) + "}]: {}").format(epoch, total_epoch_loss_test))
            epoch += 1

    # Evaluation
    y_true = evaluate_loader.labels.tolist()
    y_pred = []
    for sentence in tqdm(evaluate_loader.dataset, desc="Evaluating"):
        y_pred.append(vae.predict(sentence))

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    recall = metrics.recall_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    print()
    print('=' * 80)
    print("Results:")
    print('=' * 80)
    print()
    print("tp:", tp, "tn:", tn)
    print("fp:", fp, "fn:", fn)
    print("recall:", "{:.5f}".format(recall), "precision:", "{:.5f}".format(precision), "f1:", "{:.5f}".format(f1))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('path', metavar='PATH', help='Path to the dataset')
    arg_parser.add_argument('--seperator', metavar='SEP', default=None, help='seperator of dataset')
    arg_parser.add_argument('--epochs', metavar='E', default=100, help='Number of epochs')
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    arg_parser.add_argument('--loss-threshold', metavar='THRESHOLD', default=None,
                            help='Threshold for loss. Optimization stops prematurely if this threshold is reached.')

    args = arg_parser.parse_args()

    path = args.path
    seperator = args.seperator
    epochs = int(args.epochs)
    loss_threshold = None if args.loss_threshold is None else float(args.loss_threshold)
    use_cuda = args.cuda and torch.cuda.is_available()

    main(path, sep=seperator, epochs=epochs, loss_threshold=loss_threshold, use_cuda=use_cuda)
