import torch
from skorch import NeuralNet
from skorch.callbacks import EpochTimer, PassthroughScoring, PrintLog
from skorch.dataset import unpack_data, uses_placeholder_y, get_len, CVSplit
from skorch.history import History
from skorch.setter import optimizer_setter
from skorch.utils import to_device, to_tensor, get_map_location

from bestiary.modules.measures.critic import Wasserstein, Critic


class WassersteinGenerativeAdversarialNet(NeuralNet):
    prefixes_ = ['module', 'critic', 'iterator_train', 'iterator_valid',
                 'optimizer', 'critic_optimizer', 'callbacks', 'dataset']

    distance_ = Wasserstein

    def __init__(
            self,
            module,
            critic,
            *args,
            train_split=CVSplit(10),
            train_generator_every=1,
            critic_optimizer=torch.optim.SGD,
            **kwargs
    ):

        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.train_generator_every = train_generator_every

        super().__init__(module, *args, criterion=None, train_split=train_split, **kwargs)

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

    def initialize_criterion(self):
        pass

    def initialize_module(self):
        """Initializes the module.

        Note that if the module has learned parameters, those will be
        reset.

        """
        kwargs = self._get_params_for('module')
        kwargs_critic = self._get_params_for('critic')

        module = self.module
        critic = self.critic

        is_initialized = isinstance(module, torch.nn.Module)
        is_initialized_critic = isinstance(critic, torch.nn.Module)

        if kwargs or not is_initialized:
            if is_initialized:
                module = type(module)

            if (is_initialized or self.initialized_) and self.verbose:
                msg = self._format_reinit_msg("module", kwargs)
                print(msg)

            module = module(**kwargs)

        if kwargs_critic or not is_initialized_critic:
            if is_initialized_critic:
                critic = type(critic)

            if (is_initialized_critic or self.initialized_) and self.verbose:
                msg = self._format_reinit_msg("critic", kwargs_critic)
                print(msg)

            critic = self.distance_(critic, **kwargs_critic)

        self.module_ = to_device(module, self.device)  # type: torch.nn.Module
        self.critic_ = to_device(critic, self.device)  # type: Critic

        return self

    def initialize_optimizer(self, triggered_directly=True):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.

        Parameters
        ----------
        triggered_directly : bool (default=True)
          Only relevant when optimizer is re-initialized.
          Initialization of the optimizer can be triggered directly
          (e.g. when lr was changed) or indirectly (e.g. when the
          module was re-initialized). If and only if the former
          happens, the user should receive a message informing them
          about the parameters that caused the re-initialization.

        """
        args_gen, kwargs_gen = self._get_params_for_optimizer(
            'optimizer', self.module_.named_parameters())
        args_critic, kwargs_critic = self._get_params_for_optimizer(
            'critic_optimizer', self.critic_.named_parameters())

        if self.initialized_ and self.verbose:
            msg = self._format_reinit_msg(
                "optimizer", kwargs_gen, triggered_directly=triggered_directly)
            print(msg)
            msg = self._format_reinit_msg(
                "critic_optimizer", kwargs_critic, triggered_directly=triggered_directly)
            print(msg)

        if 'lr' not in kwargs_gen:
            kwargs_gen['lr'] = self.lr
        if 'lr' not in kwargs_critic:
            kwargs_critic['lr'] = self.lr

        # type: torch.optim.optimizer.Optimizer
        self.optimizer_ = self.optimizer(*args_gen, **kwargs_gen)
        # type: torch.optim.optimizer.Optimizer
        self.critic_optimizer_ = self.critic_optimizer(*args_critic, **kwargs_critic)

        self._register_virtual_param(
            ['optimizer__param_groups__*__*', 'optimizer__*', 'lr'],
            optimizer_setter,
        )
        self._register_virtual_param(
            ['critic_optimizer__param_groups__*__*', 'critic_optimizer__*', 'lr'],
            optimizer_setter,
        )

    def train_step(self, Xi, yi, **fit_params):
        train_generator = fit_params.pop('train_generator', True)

        self.module_.train()
        self.critic_.train()

        self.optimizer_.zero_grad()
        self.critic_optimizer_.zero_grad()

        b = Xi.shape[0]
        real = to_tensor(Xi, self.device)
        generated = self.module_.generate(b)

        critic_loss = self.critic_.loss(real, generated.detach())
        critic_loss.backward()
        self.critic_optimizer_.step()

        distance = self.critic_.distance(real, generated)

        if train_generator:
            distance.backward()
            self.optimizer_.step()

        self.history.record_batch('critic_loss', critic_loss.item())

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

    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        """Compute a single epoch of train or validation.

        Parameters
        ----------
        dataset : torch Dataset
            The initialized dataset to loop over.

        training : bool
            Whether to set the module to train mode or not.

        prefix : str
            Prefix to use when saving to the history.

        step_fn : callable
            Function to call for each batch.

        **fit_params : dict
            Additional parameters passed to the ``step_fn``.
        """
        is_placeholder_y = uses_placeholder_y(dataset)

        batch_count = 0
        for i, data in enumerate(self.get_iterator(dataset, training=training)):
            Xi, yi = unpack_data(data)
            yi_res = yi if not is_placeholder_y else None
            self.notify("on_batch_begin", X=Xi, y=yi_res, training=training)
            step = step_fn(Xi, yi, train_generator=(i % self.train_generator_every == 0), **fit_params)
            self.history.record_batch(prefix + "_distance", step["distance"].item())
            self.history.record_batch(prefix + "_batch_size", get_len(Xi))
            self.notify("on_batch_end", X=Xi, y=yi_res, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    def save_params(
            self, f_params=None, f_optimizer=None, f_history=None):
        """Saves the module's parameters, history, and optimizer,
        not the whole object.

        To save the whole object, use pickle. This is necessary when
        you need additional learned attributes on the net, e.g. the
        ``classes_`` attribute on
        :class:`skorch.classifier.NeuralNetClassifier`.

        ``f_params`` and ``f_optimizer`` uses PyTorchs'
        :func:`~torch.save`.

        Parameters
        ----------
        f_params : file-like object, str, None (default=None)
          Path of module parameters. Pass ``None`` to not save

        f_optimizer : file-like object, str, None (default=None)
          Path of optimizer. Pass ``None`` to not save

        f_history : file-like object, str, None (default=None)
          Path to history. Pass ``None`` to not save
        """
        if f_params is not None:
            msg = (
                "Cannot save parameters of an un-initialized model. "
                "Please initialize first by calling .initialize() "
                "or by fitting the model with .fit(...).")
            self.check_is_fitted(msg=msg)
            torch.save(self.module_.state_dict(), f_params)
            torch.save(self.critic_.state_dict(), f_params + '_critic')

        if f_optimizer is not None:
            msg = (
                "Cannot save state of an un-initialized optimizer. "
                "Please initialize first by calling .initialize() "
                "or by fitting the model with .fit(...).")
            self.check_is_fitted(attributes=['optimizer_'], msg=msg)
            torch.save(self.optimizer_.state_dict(), f_optimizer)
            torch.save(self.critic_optimizer_.state_dict(), f_optimizer + '_critic')

        if f_history is not None:
            self.history.to_file(f_history)

    def load_params(
            self, f_params=None, f_optimizer=None, f_history=None,
            checkpoint=None):
        """Loads the the module's parameters, history, and optimizer,
        not the whole object.

        To save and load the whole object, use pickle.

        ``f_params`` and ``f_optimizer`` uses PyTorchs'
        :func:`~torch.save`.

        Parameters
        ----------
        f_params : file-like object, str, None (default=None)
          Path of module parameters. Pass ``None`` to not load.

        f_optimizer : file-like object, str, None (default=None)
          Path of optimizer. Pass ``None`` to not load.

        f_history : file-like object, str, None (default=None)
          Path to history. Pass ``None`` to not load.

        checkpoint : :class:`.Checkpoint`, None (default=None)
          Checkpoint to load params from. If a checkpoint and a ``f_*``
          path is passed in, the ``f_*`` will be loaded. Pass
          ``None`` to not load.
        """

        def _get_state_dict(f):
            map_location = get_map_location(self.device)
            self.device = self._check_device(self.device, map_location)
            return torch.load(f, map_location=map_location)

        if f_history is not None:
            self.history = History.from_file(f_history)

        if checkpoint is not None:
            if not self.initialized_:
                self.initialize()
            if f_history is None and checkpoint.f_history is not None:
                self.history = History.from_file(checkpoint.f_history_)
            formatted_files = checkpoint.get_formatted_files(self)
            f_params = f_params or formatted_files['f_params']
            f_optimizer = f_optimizer or formatted_files['f_optimizer']

        if f_params is not None:
            msg = (
                "Cannot load parameters of an un-initialized model. "
                "Please initialize first by calling .initialize() "
                "or by fitting the model with .fit(...).")
            self.check_is_fitted(msg=msg)
            state_dict = _get_state_dict(f_params)
            state_dict_critic = _get_state_dict(f_params + '_critic')
            self.module_.load_state_dict(state_dict)
            self.critic_.load_state_dict(state_dict_critic)

        if f_optimizer is not None:
            msg = (
                "Cannot load state of an un-initialized optimizer. "
                "Please initialize first by calling .initialize() "
                "or by fitting the model with .fit(...).")
            self.check_is_fitted(attributes=['optimizer_'], msg=msg)
            state_dict = _get_state_dict(f_optimizer)
            state_dict_critic = _get_state_dict(f_optimizer + '_critic')
            self.optimizer_.load_state_dict(state_dict)
            self.critic_optimizer_.load_state_dict(state_dict_critic)
