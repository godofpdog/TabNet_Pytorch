""" Implementations of model builders. """

import os 
import abc 
import torch
import numpy as np 

from ._models import (
    TabNetEncoder, TabNetHead, EmbeddingEncoder, InferenceModel, 
    PretrainModel, TabNetPretextModel, SwapDAEPreTextModel
)


class _BaseBuilder(abc.ABC):
    def __init__(self, weights_path=None):
        self.weights_path = weights_path

    def build(self, is_cuda=False, **kwargs):
        model = self._build(**kwargs)
        self._load_weights(model)

        if is_cuda:
            model = model.cuda()

        return model 

    def _load_weights(self, model):

        if self.weights_path is not None:
            try:
                load_weights(model, self.weights_path)

            except Exception as e:
                print(e)
                self._init_weights(model)
        else:
            self._init_weights(model)

    @abc.abstractmethod
    def _build(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_weights(self, model):
        raise NotImplementedError


def load_weights(model, path):
    """
    Load model weights.

    Arguments:
        model (subclass of torch.nn.Module): Model architecture. 
        path (str): Weights path. 

    Returns:
        None
         
    """

    if not issubclass(model.__class__, torch.nn.Module):
        raise TypeError(
            'Invalid type, argument `model` must be a subclass of `torch.nn.Module`.'
        )

    model.load_state_dict(torch.load(path))

    return None 


def _init_weights(module, is_glu=False):
    # NOTE only support torch.nn.Linear

    if isinstance(module, torch.nn.Linear):
        input_dims = module.weight.size(1)
        output_dims = module.weight.size(0)

    else:
        return 

    r = 1 if is_glu else 4 

    torch.nn.init.xavier_normal_(
        module.weight, 
        gain=np.sqrt((input_dims + output_dims) / np.sqrt(r * input_dims))
    )

    return None 


class _EmbeddingEmcoderBuilder(_BaseBuilder):
    def __init__(self, weights_path=None):
        super(_EmbeddingEmcoderBuilder, self).__init__(weights_path)

    def _build(self, input_dims, cate_indices, cate_dims, embed_dims):
        return EmbeddingEncoder(
            input_dims, cate_indices, cate_dims, embed_dims
        )

    def _init_weights(self, model):
        for m in model.modules():
            _init_weights(m, is_glu=False)


class _TabNetEncoderBuilder(_BaseBuilder):
    def __init__(self, weights_path=None):
        super(_TabNetEncoderBuilder, self).__init__(weights_path)
    
    def _build(
        self, input_dims, reprs_dims=8, atten_dims=8, num_steps=3, gamma=1.3, num_indep=2, 
        num_shared=2, virtual_batch_size=128, momentum=0.02, mask_type='sparsemax'
    ):
        return TabNetEncoder(
            input_dims, reprs_dims, atten_dims, num_steps, gamma, num_indep,
            num_shared, virtual_batch_size, momentum, mask_type
        )

    def _init_weights(self, model):

        # init splitter 
        for m in model.input_splitter.modules():
            _init_weights(m, is_glu=True)

        # attentive transformers
        for m in model.atten_transformers.modules():
            _init_weights(m, is_glu=False)

        # feature transformers
        for m in model.feats_transformers.modules():
            _init_weights(m, is_glu=True)

        return None 


class _TabNetHeadBuilder(_BaseBuilder):
    def __init__(self, weights_path=None):
        super(_TabNetHeadBuilder, self).__init__(weights_path)

    def _build(self, reprs_dims, output_dims):
        return TabNetHead(
            reprs_dims, output_dims
        )

    def _init_weights(self, model):
        for m in model.modules():
            _init_weights(m, is_glu=False)


class _InferenceModelBuilder(_BaseBuilder):
    def __init__(self, weights_path=None):
        super(_InferenceModelBuilder, self).__init__(weights_path) 

    def _build(
        self, input_dims, output_dims, cate_indices, cate_dims, cate_embed_dims=1,
        reprs_dims=8, atten_dims=8, num_steps=3, gamma=1.3, num_indep=2, 
        num_shared=2, virtual_batch_size=128, momentum=0.02, mask_type='sparsemax'
    ):

        embedding_encoder = _EmbeddingEmcoderBuilder().build(
            input_dims=input_dims, cate_indices=cate_indices, cate_dims=cate_dims, embed_dims=cate_embed_dims
        )

        tabnet_encoder = _TabNetEncoderBuilder().build(
            input_dims=embedding_encoder.output_dims, reprs_dims=reprs_dims, atten_dims=atten_dims, 
            num_steps=num_steps, gamma=gamma, num_indep=num_indep, num_shared=num_shared, 
            virtual_batch_size=virtual_batch_size, momentum=momentum, mask_type=mask_type
        )

        tabnet_head = _TabNetHeadBuilder().build(reprs_dims=reprs_dims, output_dims=output_dims)

        return InferenceModel(
            embedding_encoder, tabnet_encoder, tabnet_head
        )

    def _init_weights(self, model):
        pass 


class _PretrainModelBuilder(_BaseBuilder):
    _SUPPORTED_ALGORITHM = ['tabnet']

    def __init__(self, weights_path):
        super(_PretrainModelBuilder, self).__init__(weights_path)

    def _build(self, algorithm_params):
        pass 


def build_model(model_type, weights_path=None, is_cuda=False, **kwargs):
    # TODO support `pretrain_model`

    _SUPPORTED_BUILDERS = {
        'embedding_encoder': _EmbeddingEmcoderBuilder,
        'tabnet_encoder': _TabNetEncoderBuilder,
        'tabnet_head': _TabNetHeadBuilder,
        'inference_model': _InferenceModelBuilder 
    }

    builder = _SUPPORTED_BUILDERS.get(model_type)

    if builder is not None:
        return builder(weights_path).build(is_cuda, **kwargs)
    else:
        raise ValueError('Not supported model type.')


class ModelConverter:

    @classmethod
    def to_pretrain(cls, model, algorithm_params, model_configs, device):
        """
        Convert to `PretrainModel`.

        Arguments:
            model (tabnet.core.model.PretrainModel or tabnet.core.model.InferenceModel)
                A model object.

            algorithm_params (dict):
                Parameters of the training algorithm (including the `traning algorithm`).

                - Example:
                    algorithm_params = {
                        'algorithm': 'tabnet_pretraining',
                        'mask_rate': 0.2
                    }

            model_configs (dict):
                Model architecture configurations.

            device (str):
                Computation device.

        Returns:
            (tanbet.core.PretrainModel):
                A converted model object.

        """
        model_type = model.__class__

        def _convert(model, algorithm_params, model_configs):
            embedding_encoder = getattr(model, 'embedding_encoder', None)
            tabnet_encoder = getattr(model, 'tabnet_encoder', None)

            # TODO support SwapDAE pre-training algorithm
            algorithm = algorithm_params['algorithm']

            if algorithm in ('tabnet', 'tabnet_pretraining'):
                try:
                    model_configs['mask_rate'] = algorithm_params['mask_rate']
                    pretext_model = TabNetPretextModel(**model_configs)
                except Exception as e:
                    print(e)

            elif algorithm in ('swap_dae', 'swapdae', 'swap_dae_pretraining'):
                try:
                    pretext_model = SwapDAEPreTextModel(**model_configs)
                except Exception as e:
                    print(e)
                    
            else:
                raise ValueError('Invalid training algorithm.')

            return PretrainModel(
                embedding_encoder, tabnet_encoder, pretext_model
            )

        if model_type == PretrainModel:
            return model.to(device)

        elif model_type == InferenceModel:
            return _convert(model, algorithm_params, model_configs).to(device)

        else:
            raise TypeError('Invalid model type.')

    @classmethod
    def to_inference(cls, model, model_configs, device):
        """  
        Convert to `InferenceModel`.

        Arguments:
            model (tabnet.core.model.PretrainModel or tabnet.core.model.InferenceModel)
                A model object.

            model_configs (dict):
                Model architecture configurations.

            device (str):
                Computation device.

        Returns:
            (tanbet.core.InferenceModel):
                A converted model object.

        """
        model_type = model.__class__

        def _convert(model, model_configs):
            embedding_encoder = getattr(model, 'embedding_encoder', None)
            tabnet_encoder = getattr(model, 'tabnet_encoder', None)
            tabnet_head = _TabNetHeadBuilder().build(
                reprs_dims=model_configs['reprs_dims'], output_dims=model_configs['output_dims']
            )

            return InferenceModel(
                embedding_encoder, tabnet_encoder, tabnet_head
            )
        
        if model_type == InferenceModel:
            return model.to(device)
        
        elif model_type == PretrainModel:
            return _convert(model, model_configs).to(device)

        else:
            raise TypeError('Invalid model type.')
