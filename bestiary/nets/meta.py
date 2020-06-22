from skorch import NeuralNet
from skorch.dataset import uses_placeholder_y, get_len


class MetaLearningNet(NeuralNet):

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
        for data in self.get_iterator(dataset, training=training):
            # Removes the dummy target
            data, _ = data

            # Removes the query target to limit temptation
            yi = data['query'].pop(1)
            yi_res = yi if not is_placeholder_y else None

            self.notify("on_batch_begin", X=data, y=yi_res, training=training)
            step = step_fn(data, yi, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            self.history.record_batch(prefix + "_batch_size", get_len(data))
            self.notify("on_batch_end", X=data, y=yi_res, training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)
