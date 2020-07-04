import torch
from skorch.callbacks import EpochTimer, PassthroughScoring, PrintLog
from skorch.dataset import CVSplit
from skorch.utils import to_device, to_tensor
from torch import nn

from bestiary.nets.wgan import WassersteinGenerativeAdversarialNet


class GenerativeAdversarialNet(WassersteinGenerativeAdversarialNet):
    prefixes_ = ['module', 'critic', 'iterator_train', 'iterator_valid',
                 'optimizer', 'critic_optimizer', 'callbacks', 'dataset', 'citerion']

    def __init__(
            self,
            module,
            critic,
            *args,
            criterion=nn.BCELoss,
            train_split=CVSplit(10),
            train_generator_every=1,
            critic_optimizer=torch.optim.SGD,
            **kwargs
    ):

        super().__init__(module, *args, critic=critic, critic_optimizer=critic_optimizer,
                         train_split=train_split, train_generator_every=train_generator_every, **kwargs)

        self.criterion = criterion

    def initialize_criterion(self):
        """Initializes the criterion."""
        criterion_params = self._get_params_for('criterion')
        self.criterion_ = self.criterion(**criterion_params)
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_ = to_device(self.criterion_, self.device)
        return self

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('critic_loss', PassthroughScoring(
                name='critic_loss',
                lower_is_better=True,
                on_train=True,
            )),
            ('train_distance', PassthroughScoring(
                name='train_distance',
                lower_is_better=True,
                on_train=True,
            )),
            ('valid_distance', PassthroughScoring(
                name='valid_distance',
                lower_is_better=True,
            )),
            ('inter_distance', PassthroughScoring(
                name='inter_distance',
                lower_is_better=True,
            )),
            ('print_log', PrintLog()),
        ]

    def train_step(self, Xi, yi, **fit_params):
        train_generator = fit_params.pop('train_generator', True)

        self.module_.train()
        self.critic_.train()

        self.optimizer_.zero_grad()
        self.critic_optimizer_.zero_grad()

        b = Xi.shape[0]
        real = to_tensor(Xi, self.device)
        generated = self.module_.generate(b)

        y_real = self.critic_(real)
        y_generated = self.critic_(generated.detach())

        critic_loss = self.get_loss(y_real, torch.ones_like(y_real))
        critic_loss = critic_loss + self.get_loss(y_generated, torch.zeros_like(y_generated))
        critic_loss.backward()

        self.critic_optimizer_.step()

        if train_generator:
            y_generated = self.critic_(generated)
            generator_loss = self.get_loss(y_generated, torch.ones_like(y_generated))
            generator_loss.backward()

            self.optimizer_.step()

        distance = y_real.log().mean() + (1 - y_generated).log().mean()
        distance = distance / 2

        return {
            'critic_loss': critic_loss,
            'distance': distance,
        }

    def validation_step(self, Xi, yi, **fit_params):
        """Perform a forward step using batched data and return the
        resulting loss.

        The module is set to be in evaluation mode (e.g. dropout is
        not applied).

        Parameters
        ----------
        Xi : input data
          A batch of the input data.

        yi : target data
          A batch of the target data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self.module_.eval()
        self.critic_.eval()

        real = to_tensor(Xi, self.device)
        b = real.shape[0]

        with torch.no_grad():
            real1, real2 = torch.chunk(real, 2)

            inter_distance = self.critic_.distance(real1, real2)
            distance = self.critic_.distance(real, self.module_.generate(b))

        self.history.record_batch('inter_distance', inter_distance)

        return {
            'inter_distance': inter_distance,
            'distance': distance,
        }

    def generate(self, n=20):
        self.module_.eval()
        with torch.no_grad():
            return self.module_.generate(n).numpy()

    def distance(self, real):
        real = to_tensor(real, self.device)
        self.module_.eval()
        self.critic_.eval()
        with torch.no_grad():
            fake = self.module_.generate(real.shape[0])
            return self.critic_.distance(real, fake).item()
