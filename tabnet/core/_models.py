""" Implementatoins of TabNet model architectures. """

import abc
import torch
import pickle
import numpy as np
import torch.nn as nn
from ._masks import Sparsemax, Entmax15


_EPSILON = 1e-15


class GhostBatchNorm(nn.Module):
    """
    Implementation of the Ghost Batch Normalization 
    as described in the paper: https://arxiv.org/abs/1705.08741
    """
    def __init__(self, input_dims, virtual_batch_size, momentum=0.01):
        super(GhostBatchNorm, self).__init__()
        """
        Initialization of `GhostBatchNorm` module.

        Arguments:
            input_dims (int): Dimension of input features.
            virtual_batch_size (int): Virtual batch size.
            momentum (float): Momentum parameters.

        Returns:
            None

        """
        self.input_dims = input_dims
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dims, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(x.size(0) // self.virtual_batch_size, 0)
        return torch.cat([self.bn(x_) for x_ in chunks], dim=0)


class GLUBlock(nn.Module):
    """
    Implementation of `GLUBlock` which contains fully-connected layer, 
    Ghost Batch Normalization layer and a Gate Linear Unit architectires. 
    """
    def __init__(self, input_dims, output_dims, share_layer, virtual_batch_size, momentum):
        super(GLUBlock, self).__init__()
        """
        Initialization of `GLUBlock` module.

        Arguments:
            input_dims (int): Dimension of input features. 
            output_dims (int): Dimension of output features. 
            shared_layer (torch.nn.Linear): Shared fully-connected layer cross all steps.
            virtual_batch_size (int): Virtual batch size in `GhostBatchNorm` module. 
            momentum (float): Momentum parameters in `GhostBatchNorm` module. 
        
        Returns:
            None

        """
        self.output_dims = output_dims

        if share_layer is not None:
            self.fc = share_layer
        else:
            self.fc = nn.Linear(input_dims, output_dims * 2, bias=False)

        self.gbn = GhostBatchNorm(output_dims * 2, virtual_batch_size, momentum)

    def forward(self, x):
        x = self.gbn(self.fc(x))
        return torch.mul(x[:, :self.output_dims], torch.sigmoid(x[:, self.output_dims:])) 


class FeatureBlock(nn.Module):
    """
    Computational block for feature extraction, contains `num_glu` of `GLUBlock`.
    """
    def __init__(
        self, input_dims, output_dims, shared_layers=None, num_glu=2, 
        is_first=False, virtual_batch_size=128, momentum=0.02
    ):
        """
        Initialization if `FeatureBlock` module.

        Arguments:
            input_dims (int): Dimension of input features. 
            output_dims (int): Dimension of output features. 
            shared_layers (torch.nn.Linear): Shared fully-connected layers cross all steps.
            num_glu (int): Number of `GLUBlock` in the module.
            is_first (bool): If True, means that this module is the first layer of the TabNet model. (different `inout_dims` in `GLUBlock`).
            virtual_batch_size (int): Virtual batch size in `GhostBatchNorm` module. 
            momentum (float): Momentum parameters in `GhostBatchNorm` module. 

        Returns:
            None

        """
        super(FeatureBlock, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.shared_layers = shared_layers
        self.num_glu = num_glu
        self.is_first = is_first
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self._build()

    def _build(self):
        self.glu_blocks = nn.ModuleList()

        for i in range(self.num_glu):

            if i == 0:
                input_dims = self.input_dims
            else:
                input_dims = self.output_dims
            
            if self.shared_layers is not None:
                shared_layer = self.shared_layers[i]
            else:
                shared_layer = None 

            self.glu_blocks.append(
                GLUBlock(input_dims, self.output_dims, shared_layer, self.virtual_batch_size, self.momentum)
            )

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))

        if self.is_first:
            x = self.glu_blocks[0](x)
            s = 1
        else:
            s = 0
        
        for i in range(s, self.num_glu):
            x = x + self.glu_blocks[i](x)
            x = x * scale
        
        return x


class AttentiveTransformer(nn.Module):
    """
    Implementation of `AttentiveTransformer` which conbines processed features from previous `FeatureTransformer` 
    and the prior information to obtain the soft-mask for salient feature selection.
    """
    def __init__(self, input_dims, output_dims, virtual_batch_size=128, momentum=1.3, mask_type='sparsemax'):
        super(AttentiveTransformer, self).__init__()
        """
        Initialization of `AttentiveTransformer` module.

        Arguments:
            input_dims (int): Dimension of input features from `FeatureTransformer`. 
            output_dims (int): Dimension of output mask, equal to the dimension of input features (number of columns in tabular data). 
            virtual_batch_size (int):  Virtual batch size in `GhostBatchNorm` module. 
            momentum (float): Momentum parameters in `GhostBatchNorm` module. 
            mask_type (str): Mask type in `AttentiveTransformer`. 
        
        Returns:
            None

        """
        self.fc = nn.Linear(input_dims, output_dims, bias=False)
        self.gbn = GhostBatchNorm(output_dims, virtual_batch_size, momentum)

        if mask_type == 'sparsemax':
            self.selector = Sparsemax(dim=-1) 
        elif mask_type == 'entmax':
            self.selector = Entmax15(dim=-1)
        else:
            raise NotImplementedError

    def forward(self, p, x):
        x = self.fc(x)
        x = self.gbn(x)
        x = torch.mul(x, p)
        x = self.selector(x)
        return x


class FeatureTransformer(nn.Module):
    """
    Implementation of TabNet Feature Transformer which contains two `FeatureBlock` for feature extraction 
    and split input features to 2 branches (one for decision representation, one for attentive masking).
    """
    def __init__(self, input_dims, output_dims, shared_layers, num_indep, virtual_batch_size=128, momentum=0.02):
        super(FeatureTransformer, self).__init__()
        """
        Initialization of `FeatureTransformer` module.

        Arguments:
            input_dims (int): Dimension of input features. 
            output_dims (int): Dimension of output features. 
            shared_layers (torch.nn.Linear): Shared fully-connected layers cross all steps 
            num_indep (int): Number of step-specified `GLUBlock` in each `FeatureTransformer`. 
            virtual_batch_size (int): Virtual batch size in `GhostBatchNorm` module. 
            momentum (float): Momentum parameters in `GhostBatchNorm` module. 

        Returns:
            None

        """
        is_first = True
        
        if shared_layers is None:
            self.shared_block = torch.nn.Identity()
        else:
            self.shared_block = FeatureBlock(
                input_dims, output_dims, shared_layers, len(shared_layers), is_first, virtual_batch_size, momentum
            )

            is_first = False

        if num_indep == 0:
            self.indep_block = torch.nn.Identity()
        else:
            indep_input_dims = input_dims if is_first else output_dims
            self.indep_block = FeatureBlock(
                indep_input_dims, output_dims, None, num_indep, is_first, virtual_batch_size, momentum
            )

    def forward(self, x):
        return self.indep_block(self.shared_block(x))


class TabNetEncoder(nn.Module):
    """
    Implementation of the TabNet Encoder for feature extraction.
    """
    def __init__(
        self, input_dims, reprs_dims=8, atten_dims=8, num_steps=3, gamma=1.3, 
        num_indep=2, num_shared=2, virtual_batch_size=128, momentum=0.02, mask_type='sparsemax'
    ):
        """ 
        Initialization of the `TabNetEncoder` module.

        Arguments:
            input_dims (int): Dimension of input features from `EmbeddingEncoder`. 
            reprs_dims (int): Dimension of decision representaion. 
            atten_dims (int): Dimension of attentive features. 
            num_steps (int): Number of decision steps. 
            gamma (float): Scaling factor for attention updates 
            num_indep (int): Number of step-specified `GLUBlock` in each `FeatureTransformer`. 
            num_shared (int): Number of shared fully-connected layers cross all steps. 
            virtual_batch_size (int): Virtual batch size in `GhostBatchNorm` module. 
            momentum (float): Momentum parameters in `GhostBatchNorm` module. 
            mask_type (str): Mask type in `AttentiveTransformer`. 

        Returns:
            None

        """
        super(TabNetEncoder, self).__init__()
        self.input_dims = input_dims
        self.reprs_dims = reprs_dims
        self.atten_dims = atten_dims
        self.num_steps = num_steps
        self.gamma = gamma
        self.num_indep = num_indep
        self.num_shared = num_shared
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self._build()

    def _build(self):
        """
        build tabnet encoder architecture.
        """
        hidden_dims = self.reprs_dims + self.atten_dims

        # build shared layers
        if self.num_shared > 0:
            shared_layers = nn.ModuleList()
            
            for i in range(self.num_shared):

                if i == 0:
                    shared_layers.append(
                        nn.Linear(self.input_dims, hidden_dims * 2, bias=False)
                    )
                else:
                    shared_layers.append(
                        nn.Linear(hidden_dims, hidden_dims * 2, bias=False)
                    )
        else:
            shared_layers = None 

        # build main layers
        self.input_bn = nn.BatchNorm1d(self.input_dims, momentum=self.momentum)

        self.input_splitter = FeatureTransformer(
            self.input_dims, hidden_dims, shared_layers, self.num_indep, self.virtual_batch_size, self.momentum
        )

        self.feats_transformers = nn.ModuleList()
        self.atten_transformers = nn.ModuleList()

        for _ in range(self.num_steps):
            self.feats_transformers.append(
                FeatureTransformer(
                    self.input_dims, hidden_dims, shared_layers, self.num_indep, self.virtual_batch_size, self.momentum
                )
            )

            self.atten_transformers.append(
                AttentiveTransformer(
                    self.atten_dims, self.input_dims, self.virtual_batch_size, self.momentum, self.mask_type
                )
            )

    def forward(self, x, is_explain=False):
        """
        Define forward computation.

        Arguments:
            x (Tensor): Input tensor.
            is_explain (bool): If True, return interpretive infomation.

        Returns:
            outputs (list of Tensors): Decision representations of each steps.
            m_loss (Tensor): The Mask loss.
            m_explain (Tensor): Mask interpretive infomation.
            masks (dict of Tensor): Attentive mask of each steps.

        """
        m_loss = 0
        outputs = []

        prior = torch.ones_like(x).to(x.device)

        if is_explain:
            m_explain = torch.zeros(x.shape).to(x.device)
            masks = dict()

        x = self.input_bn(x)
        atten = self.input_splitter(x)[:, self.reprs_dims:]

        for step in range(self.num_steps):
            m = self.atten_transformers[step](prior, atten)
            m_loss += torch.mean(
                torch.sum(torch.mul(m, torch.log(m + _EPSILON)), dim=1)
            )
            prior = torch.mul(self.gamma - m, prior)
            output = self.feats_transformers[step](torch.mul(m, x))
            feats = nn.ReLU()(output[:, :self.reprs_dims])
            outputs.append(feats)
            atten = output[:, self.reprs_dims:]

            if is_explain:
                masks[step] = m
                step_contrib = torch.sum(feats, dim=1)
                m_explain += torch.mul(m, step_contrib.unsqueeze(dim=1))
        
        m_loss /= self.num_steps 

        if is_explain:
            return outputs, m_loss, m_explain, masks

        return outputs, m_loss

    def explain(self, x):
        """
        Compute interpretive infomation

        Arguments:
            x (Tensor): Input tensor.

        Returns:
            m_explain (Tensor): Mask interpretive infomation.
            masks (dict of Tensor): Attentive mask of each steps.

        """
        prior = torch.ones_like(x).to(x.device)
        m_explain = torch.zeros(x.shape).to(x.device)
        masks = dict()

        x = self.input_bn(x)
        atten = self.input_splitter(x)[:, self.reprs_dims:]

        for step in range(self.num_steps):
            m = self.atten_transformers[step](prior, atten)
            prior = torch.mul(self.gamma - m, prior)
            output = self.feats_transformers[step](torch.mul(m, x))
            atten = output[:, self.reprs_dims:]
            feats = nn.ReLU()(output[:, :self.reprs_dims])
            masks[step] = m
            step_contrib = torch.sum(feats, dim=1)
            m_explain += torch.mul(m, step_contrib.unsqueeze(dim=1))
        
        return m_explain, masks


class TabNetHead(nn.Module):
    """
    Implementation of tabnet for the downstream tasks. Multi-task is available .
    """
    def __init__(self, reprs_dims, output_dims):
        super(TabNetHead, self).__init__()
        """
        Initialization of `TabNetHead` module.

        Arguments:
            reprs_dims (int): Dimension of decision representation. 
            output_dims (list or int): Output dimensions, list of dims means apply multi-task. 

        Returns:
            None

        """
        self.reprs_dims = reprs_dims
        self.output_dims = output_dims 
        self._build()

    def _build(self):
        self.heads = nn.ModuleList()

        if isinstance(self.output_dims, int):
            self.heads.append(
                nn.Linear(self.reprs_dims, self.output_dims, bias=False)
            )

        else:
            for dims in self.output_dims:
                self.heads.append(
                    nn.Linear(self.reprs_dims, dims, bias=False)
                )
    
    def forward(self, x):
        res = []

        for head in self.heads:
            res.append(head(x))
        return res


class TabNetDecoder(nn.Module):
    def __init__(
        self, input_dims, reprs_dims=8, num_steps=3, num_indep=2, 
        num_shared=2, virtual_batch_size=128, momentum=0.02
    ):
        super(TabNetDecoder, self).__init__()
        self.input_dims = input_dims
        self.reprs_dims = reprs_dims
        self.num_steps = num_steps
        self.num_indep = num_indep
        self.num_shared = num_shared
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self._build()

    def _build(self):
        self.feats_transformers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # build shared layers
        if self.num_shared > 0:
            shared_layers = nn.ModuleList()

            for _ in range(self.num_shared):
                shared_layers.append(
                    nn.Linear(self.reprs_dims, self.reprs_dims * 2, bias=False)
                )

        else:
            shared_layers = None 

        # build step-specified layers
        for _ in range(self.num_steps):
            self.feats_transformers.append(
                FeatureTransformer(self.reprs_dims, self.reprs_dims, shared_layers, self.num_indep, self.virtual_batch_size, self.momentum)
            )

            self.fc_layers.append(
                nn.Linear(self.reprs_dims, self.input_dims, bias=False)
            )

    def forward(self, x):
        """
        Define forward computation.

        Arguments:
            x (list of Tensor):
                The output tensor from `TabNetEncoder` (outputs)

        Returns:
            r (Tensor)
                Reconstruction with shape = (batch_size, input_dims)

        """
        r = 0

        for step in range(self.num_steps):
            o = self.feats_transformers[step](x[step])
            o = self.fc_layers[step]
            r = torch.add(o, r)
        
        return r


class EmbeddingEncoder(nn.Module):
    """
    Implementation of Embedding Encoder for simple data pre-processing. 
    """
    def __init__(self, input_dims, cate_indices, cate_dims, embed_dims):
        super(EmbeddingEncoder, self).__init__()
        """
        Initialization of `EmbeddingEncoder` module.

        Arguments:
            input_dims (int): 
                Dimension of input raw features. 
            
            cate_indices (list of int or int): 
                Indices of categorical features. 
            
            cate_dims (list of int or int): 
                Number of categories in each categorical features. 
            
            embed_dims (list of int or int): 
                Dimensions of representation of embedding layer. 
        
        Returns:
            None

        """
        self._is_skip = False 

        if cate_indices is None:
            cate_indices = []

        if cate_dims is None:
            cate_dims = []

        if embed_dims is None:
            embed_dims = 1
        
        if isinstance(cate_indices, int):
            cate_indices = [cate_indices]

        if isinstance(cate_dims, int):
            cate_dims = [cate_dims]

        if cate_indices == [] or cate_dims == []:
            self._is_skip = True 
            self.output_dims = input_dims
            return 

        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(cate_indices)

        if len(cate_indices) != len(embed_dims):
            raise ValueError('`cate_indices` and `embed_dims` must have same length, but got {} and {}.'\
                .format(len(cate_indices), len(embed_dims)))
        
        self.sorted_indices = np.argsort(cate_indices)
        self.cate_indices = [cate_indices[i] for i in self.sorted_indices]
        self.cate_dims = [cate_dims[i] for i in self.sorted_indices]
        self.embed_dims = [embed_dims[i] for i in self.sorted_indices]
        self.output_dims = int(input_dims + np.sum(embed_dims) - len(embed_dims))

        # build models
        self.embedding_layers = nn.ModuleList()
    
        for cate_dim, embed_dim in zip(self.cate_dims, self.embed_dims):
            self.embedding_layers.append(
                nn.Embedding(cate_dim, embed_dim)
            )

        # conti indices
        self.conti_indices = torch.ones(input_dims, dtype=torch.bool)
        self.conti_indices[self.cate_indices] = 0

    def forward(self, x):
        outputs = []
        cnt = 0

        if self._is_skip:
            return x

        for i, is_conti in enumerate(self.conti_indices):

            if is_conti:
                outputs.append(
                    x[:, i].float().view(-1, 1)
                )

            else:
                outputs.append(
                    self.embedding_layers[cnt](x[:, i].long())
                )

                cnt +=1

        return torch.cat(outputs, dim=1)
            

class InferenceModel(nn.Module):
    """
    Implementation of Inference Model which contain three sub-modules:
        (1) `EmbeddingEncoder` for categorical features preprocessing.
        (2) `TabNetEncoder` for feature extraction.
        (3) `TabNetHead` for the specific tasks.  

    """
    def __init__(self, embedding_encoder, tabnet_encoder, tabnet_head):
        super(InferenceModel, self).__init__()
        # TODO check inputs
        self.embedding_encoder = embedding_encoder 
        self.tabnet_encoder = tabnet_encoder
        self.tabnet_head = tabnet_head

    def forward(self, x, is_explain=False):
        """
        Define forward computation.

        Arguments:
            x (Tensor): Input tensor.
            is_explain (bool): If True, return interpretive infomation.

        Returns:
            outputs (Tensors): Outputs of the inference model.
            m_loss (Tensor): The Mask loss.
            m_explain (Tensor): Mask interpretive infomation.
            masks (dict of Tensor): Attentive mask of each steps.

        """
        x = self.embedding_encoder(x)

        if is_explain:
            d_reprs, m_loss, m_explain, masks = \
                self.tabnet_encoder(x)
        else:
            d_reprs, m_loss = self.tabnet_encoder(x)

        outputs = self.tabnet_head(
            torch.sum(torch.stack(d_reprs, dim=0), dim=0)
        )

        if is_explain:
            return outputs, m_loss, m_explain, masks

        return outputs, m_loss

    def explain(self, x):
        x = self.embedding_encoder(x)
        return self.tabnet_encoder.explain(x)


class _BasePretextModel(nn.Module, abc.ABC):
    """
    Base Class of pretext task model.
    """
    def __init__(self, encode_dims):
        pass 
    
    @abc.abstractmethod
    def pre_process(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(self, x):
        raise NotImplementedError


class TabNetPretextModel(_BasePretextModel):
    """
    Implementation of default pretext task model for 
    sulf-supervised pre-training described in TabNet paper.

    The `TabNetPretextModel` module contains two sub-modules:
        (1) A `BinaryMasker` module to generate pretext task target and the initial prior.
        (2) A `TabNetDecoder` module for the reconstruction task.

    """
    def __init__(
        self, mask_rate=0.2, input_dims, reprs_dims=8, num_steps=3, 
        num_indep=2, num_shared=2, virtual_batch_size=128, momentum=0.02
    ):
        """
        Initialization of `TabNetPretextModel` module.

        Arguments:
            mask_rate (float):
                xx

            

        """
        super(TabNetPretextModel, self).__init__()
        self.masker = BinaryMasker(mask_rate)
        self.decoder = TabNetDecoder(
            input_dims, reprs_dims, num_steps, num_indep, num_shared, virtual_batch_size, momentum
        )

    def forward(self, x):
        return 

    def pre_process(self, x):
        return self.masker(x)

    def post_process(self, x):
        return self.decoder(x)


class BinaryMasker(nn.Module):
    def __init__(self, mask_rate):
        super(BinaryMasker, self).__init__()
        self.mask_rate = mask_rate

    def forward(self, x):
        """
        Define the forward computation.

        Arguments:
            x (torch.Tensor):
                The embedded features. (output of `EmbeddingEncoder`)
        
        Returns:
            masked_x (torch.Tensor):
                The mask features.

            mask (torch.Tensor):
                The binary mask, used to calc init piror and get target for pre-training.

        """
        mask = torch.bernoulli(self.mask_rate * torch.ones(x.shape)).to(x.device)
        masked_x = torch.mul(1 - mask, x)

        return masked_x, mask


class PretrainModel(nn.Module):
    """
    Implementation of the pre-train model for encoder model pre-training.

    The `PretrainModel` module contain three sub-modules:
        (1) `EmbeddingEncoder` for categorical features preprocessing.
        (2) `TabNetEncoder` for feature extraction.
        (3) `PretextMpdel` for self-supervised learning. (subclass of `_BasePretextModel`)

    `PretrainModel`-to-`InferenceModel` conversion is available by calling `convert_model` function.
    
    """
    def __init__(self, embedding_encoder, tabnet_encoder, pretext_model):
        super(PretrainModel, self).__init__()
        """
        Initialization of `PretrainModel` module.

        Arguments:
            embedding_encoder (tabnet.core._models.EmbeddingEncoder):
                A `EmbeddingEncoder` object for categorical features preprocessing.

            tabnet_encoder (tabnet.core._models.TabNetEncoder):
                A `TabNetEncoder` object for feature extraction.

            pretext_model (subclass of tabnet.core._models._BasePretextModel)
                A pretext task model for sppecified SSL pretraining approach.

        Returns:
            None

        """
        if not isinstance(embedding_encoder, EmbeddingEncoder):
            raise TypeError('Argument `embedding_encoder` must be a `EmbeddingEncoder`, but got `{}`.'.format(type(embedding_encoder)))

        if not isinstance(tabnet_encoder, TabNetEncoder):
            raise TypeError('Argument `tabnet_encoder` must be a `TabNetEncoder`, but got `{}`.'.format(type(tabnet_encoder)))

        if not issubclass(pretext_model, _BasePretextModel):
            raise TypeError('Class of argument `pretext_model` must be subclass of `_BasePretextModel`')

        self.embedding_encoder = embedding_encoder
        self.tabnet_encoder = tabnet_encoder
        self.pretext_model = pretext_model


    def forward(self, x):
        """
        Define forward computation.

        """
        x = self.embedding_encoder(x)
        encoded_x, m_loss = self.tabnet_encoder(x)
        outputs = self.pretext_model.post_process()
        