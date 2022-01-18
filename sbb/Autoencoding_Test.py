import random
from inspect import isclass

import pyro.distributions as dist
import torch
from pyro.distributions.util import broadcast_shape
from pyro.infer import config_enumerate, SVI, Trace_ELBO
from pyroapi import pyro
from torch import nn
from tqdm.notebook import trange

# %%
words = ["bad{}".format(i + 1) for i in range(5)] + ["good{}".format(i + 1) for i in range(5)]

# %%
sentences = []
for i in range(10):
    length = torch.distributions.LogNormal(1., .5).sample().int().item() * 3
    sentences.append(random.choices(words, k=length))


# %%
def label(sentence):
    bads = 0
    for word in sentence:
        if word.startswith('b'):
            bads += 1

    return int(bads > (len(sentence) - bads))


# %%
labels = [label(sentence) for sentence in sentences]

# %%
vocabulary = dict()
vec_len = max(len(sentence) for sentence in sentences)
counter = 1
sentences_vec = []
for sentence in sentences:
    vec = [0 for _ in range(vec_len)]
    for i, word in enumerate(sentence):
        if word not in vocabulary:
            vocabulary[word] = counter
            counter += 1
        vec[i] = vocabulary[word]
    sentences_vec.append(vec)



# %%
class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)


# %%
class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """

    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            # regardless of type,
            # we don't care about single objects
            # we just index into the object
            input_args = input_args[0]

        # don't concat things that are just single objects
        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)


# %%
class ListOutModule(nn.ModuleList):
    """
    a custom module for outputting a list of tensors from a list of nn modules
    """

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        # loop over modules in self, apply same args
        return [mm.forward(*args, **kwargs) for mm in self]


def call_nn_op(op):
    """
    a helper function that adds appropriate parameters when calling
    an nn module representing an operation like Softmax
    :param op: the nn.Module operation to instantiate
    :return: instantiation of the op module with appropriate parameters
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()


# %%
class MLP(nn.Module):
    def __init__(
            self,
            mlp_sizes,
            activation=nn.ReLU,
            output_activation=None,
            post_layer_fct=lambda layer_ix, total_layers, layer: None,
            post_act_fct=lambda layer_ix, total_layers, layer: None,
            allow_broadcast=False,
            use_cuda=False,
    ):
        # init the module object
        super().__init__()

        assert len(mlp_sizes) >= 2, "Must have input and output layer sizes defined"

        # get our inputs, outputs, and hidden
        input_size, hidden_sizes, output_size = (
            mlp_sizes[0],
            mlp_sizes[1:-1],
            mlp_sizes[-1],
        )

        # assume int or list
        assert isinstance(
            input_size, (int, list, tuple)
        ), "input_size must be int, list, tuple"

        # everything in MLP will be concatted if it's multiple arguments
        last_layer_size = input_size if isinstance(input_size, int) else sum(input_size)

        # everything sent in will be concatted together by default
        all_modules = [ConcatModule(allow_broadcast)]

        # loop over l
        for layer_ix, layer_size in enumerate(hidden_sizes):
            assert isinstance(layer_size, int), "Hidden layer sizes must be ints"

            # get our nn layer module (in this case nn.Linear by default)
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)

            # for numerical stability -- initialize the layer properly
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)

            # use GPUs to share data during training (if available)
            if use_cuda:
                cur_linear_layer = nn.DataParallel(cur_linear_layer)

            # add our linear layer
            all_modules.append(cur_linear_layer)

            # handle post_linear
            post_linear = post_layer_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )

            # if we send something back, add it to sequential
            # here we could return a batch norm for example
            if post_linear is not None:
                all_modules.append(post_linear)

            # handle activation (assumed no params -- deal with that later)
            all_modules.append(activation())

            # now handle after activation
            post_activation = post_act_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )

            # handle post_activation if not null
            # could add batch norm for example
            if post_activation is not None:
                all_modules.append(post_activation)

            # save the layer size we just created
            last_layer_size = layer_size

        # now we have all of our hidden layers
        # we handle outputs
        assert isinstance(
            output_size, (int, list, tuple)
        ), "output_size must be int, list, tuple"

        if isinstance(output_size, int):
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(
                    call_nn_op(output_activation)
                    if isclass(output_activation)
                    else output_activation
                )
        else:

            # we're going to have a bunch of separate layers we can spit out (a tuple of outputs)
            out_layers = []

            # multiple outputs? handle separately
            for out_ix, out_size in enumerate(output_size):

                # for a single output object, we create a linear layer and some weights
                # we have an activation function
                split_layer = [nn.Linear(last_layer_size, out_size)]

                # then we get our output activation (either we repeat all or we index into a same sized array)
                act_out_fct = (
                    output_activation
                    if not isinstance(output_activation, (list, tuple))
                    else output_activation[out_ix]
                )

                if act_out_fct:
                    # we check if it's a class. if so, instantiate the object
                    # otherwise, use the object directly (e.g. pre-instaniated)
                    split_layer.append(
                        call_nn_op(act_out_fct) if isclass(act_out_fct) else act_out_fct
                    )

                # our outputs is just a sequential of the two
                out_layers.append(nn.Sequential(*split_layer))

            all_modules.append(ListOutModule(out_layers))

        # now we have all of our modules, we're ready to build our sequential!
        # process mlps in order, pretty standard here
        self.sequential_mlp = nn.Sequential(*all_modules)

    # pass through our sequential for the output!
    def forward(self, *args, **kwargs):
        return self.sequential_mlp.forward(*args, **kwargs)


# %%
class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset
    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """

    def __init__(
            self,
            output_size=2,
            input_size=vec_len,
            z_dim=vec_len,
            hidden_layers=(vec_len,),
            config_enum=None,
            use_cuda=False,
            aux_loss_multiplier=None,
    ):

        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == "parallel"
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):

        z_dim = self.z_dim
        if isinstance(self.hidden_layers, list):
            hidden_sizes = self.hidden_layers
        elif isinstance(self.hidden_layers, int):
            hidden_sizes = [self.hidden_layers]
        else:
            hidden_sizes = [size for size in self.hidden_layers]

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter
        self.encoder_y = MLP(
            [self.input_size] + hidden_sizes + [self.output_size],
            activation=nn.Softplus,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        self.encoder_z = MLP(
            [self.input_size + self.output_size] + hidden_sizes + [z_dim],
            activation=nn.Softplus,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        self.decoder = MLP(
            [z_dim + self.output_size] + hidden_sizes + [self.input_size],
            activation=nn.Softplus,
            output_activation=nn.Sigmoid,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # handwriting style (latent)
        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
        p(x|y,z) = bernoulli(loc(y,z))   # an image
        loc is given by a neural network  `decoder`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate("data"):
            # sample the handwriting style from the constant prior distribution
            prior_logits = torch.zeros(batch_size, self.z_dim, **options)
            zs = pyro.sample("z", dist.RelaxedOneHotCategoricalStraightThrough(logits=prior_logits).to_event(1))

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            ys = pyro.sample("y", dist.RelaxedBernoulliStraightThrough(torch.tensor(0.5)), obs=ys)

            # Finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network. We disable validation
            # since the decoder output is a relaxed Bernoulli value.
            logits = self.decoder.forward([zs, ys])
            pyro.sample(
                "x", dist.Categorical(logits=logits, validate_args=False).to_event(1), obs=xs
            )
            # return the loc so we can visualize it later
            return logits

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                p = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.RelaxedBernoulliStraightThrough(p))

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            logits = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.RelaxedOneHotCategoricalStraightThrough(logits).to_event(1))


# %%
# batch_size: number of images (and labels) to be considered in a batch
ss_vae = SSVAE(
    input_size=vec_len,
    z_dim=vec_len,
    hidden_layers=vec_len,
    use_cuda=False,
    config_enum="parallel",
)

# setup the optimizer
adam_params = {"lr": 0.025, "betas": (0.95, 0.999)}
optimizer = pyro.optim.Adam(adam_params)

# set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
# by enumerating each class label for the sampled discrete categorical distribution in the model
guide = config_enumerate(ss_vae.guide, "parallel", expand=True)
Elbo = Trace_ELBO
elbo = Elbo(max_plate_nesting=1, strict_enumeration_warning=False)
loss_basic = SVI(ss_vae.model, guide, optimizer, loss=elbo)

# build a list of all losses considered
losses = [loss_basic]


# %%
def run_inference_for_epoch(xs, ys, losses):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)
    epoch_losses = []

    # count the number of supervised batches seen in this epoch

    # run the inference for each loss with supervised or un-supervised
    # data as arguments
    for loss_id in range(num_losses):
        new_loss = losses[loss_id].step(xs, ys)
        epoch_losses[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses


# %%
for _ in trange(5):
    epoch_losses = run_inference_for_epoch(torch.tensor(sentences_vec), torch.tensor(labels), losses)
