""" Implementations of estimators. """

from ..core.base import TabNetBase
from ..core.criterion import create_criterion


class TabNetRegressor(TabNetBase):
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3,
        cate_indices=None, cate_dims=None, cate_embed_dims=1, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        mask_type='sparsemax', task_weights=1, is_shuffle=True, num_workers=4, pin_memory=True, device=None, logger=None
    ):
        super(TabNetRegressor, self).__init__(
            input_dims, output_dims, reprs_dims, atten_dims, num_steps, num_indep, num_shared, gamma, 
            cate_indices, cate_dims, cate_embed_dims, batch_size, virtual_batch_size, momentum,
            mask_type, is_shuffle, num_workers, pin_memory, device, logger
        )
        """
        Initialization of `TabNetRegressor`.

        Arguments:
            xxx

        Returns:
            xxx

        """
        self.task_weights = task_weights

        self._create_criterion()

    def _create_criterion(self):
        """
        Create default criterions of regression task.
        """
        if not isinstance(self.output_dims, (int, list)):
            raise TypeError(
                'Argument `output_dim` must be a `int` or `list of int` object, but got `{}`'\
                .format(type(self.output_dims))
            )

        return create_criterion(
            task_types='regression', logits_dims=self.output_dims, weights=self.task_weights
        )
        

