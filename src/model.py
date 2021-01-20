""" Implementatoins of TabNet model architectures. """

import torch
import pickle
import torch.nn as nn
from .sparsemax import Sparsemax, Entmax15


_EPSILON = 1e-15


class GhostBatchNorm(nn.Module):
    def __init__(self, input_dims, virtual_batch_size, momentum=0.01):
        super(GhostBatchNorm, self).__init__()
        self.input_dims = input_dims
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dims, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(x.size(0) // self.virtual_batch_size, 0)
        return torch.cat([self.bn(x_) for x_ in chunks], dim=0)


class GLUBlock(nn.Module):
    def __init__(self, input_dims, output_dims, share_layer, virtual_batch_size, momentum):
        super(GLUBlock, self).__init__()
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
    def __init__(self, input_dims, output_dims, shared_layers=None, num_glu=2, is_first=False, virtual_batch_size=128, momentum=0.02):
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
        :params input_dims: Dimension of input features from `FeatureTransformer`. (int)
        :params output_dims: Dimension of output mask, equal to the dimension of input features (number of columns in tabular data). (int)
        :params virtual_batch_size:  Virtual batch size in `GhostBatchNorm` module. (int)
        :params momentum: Momentum parameters in `GhostBatchNorm` module. (float)
        :params mask_type: Mask type in `AttentiveTransformer`. (str)
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
        :params input_dims: Dimension of input features. (int)
        :params output_dims: Dimension of output features. (int)
        :params shared_layers: Shared fully-connected layers cross all steps (torch.nn.Linear)
        :params num_indep: Number of step-specified `GLUBlock` in each `FeatureTransformer`. (int)
        :params virtual_batch_size: Virtual batch size in `GhostBatchNorm` module. (int)
        :params momentum: Momentum parameters in `GhostBatchNorm` module. (float)
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
    def __init__(
        self, input_dims, reprs_dims=8, atten_dims=8, num_steps=3, gamma=1.3, 
        num_indep=2, num_shared=2, virtual_batch_size=128, momentum=0.02, mask_type='sparsemax'):
        """ 
        Implementation of the TabNet Encoder.
        :params input_dims: Dimension of input features. (int)
        :params reprs_dims: Dimension of decision representaion. (int)
        :params atten_dims: Dimension of attentive features. (int)
        :params num_steps: Number of decision steps. (int)
        :params gamma: Scaling factor for attention updates (float)
        :params num_indep: Number of step-specified `GLUBlock` in each `FeatureTransformer`. (int)
        :params num_shared: Number of shared fully-connected layers cross all steps. (int)
        :params virtual_batch_size: Virtual batch size in `GhostBatchNorm` module. (int)
        :params momentum: Momentum parameters in `GhostBatchNorm` module. (float)
        :params mask_type: Mask type in `AttentiveTransformer`. (str)
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

    def forward(self, x, prior=None):
        m_loss = 0
        outputs = []

        if prior is None:
            prior = torch.ones_like(x).to(x.device)

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
        
        m_loss /= self.num_steps 

        return outputs, m_loss

    def forward_mask(self):
        return 
        
    
class TabNetHead(nn.Module):
    def __init__(self, reprs_dims, output_dims):
        super(TabNetHead, self).__init__()
        """
        Implementation of tabnet for the downstream tasks. Multi-task is avariable.
        :params: reprs_dims: Dimension of decision representation. (int)
        :params: output_dims: Output dimensions, list of dims means apply multi-task. (list or int)
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
    def __init__(self, input_dims, reprs_dims=8, num_steps=3, num_indep=2, num_shared=2, virtual_batch_size=128, momentum=0.02):
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
        self.dense_layers = nn.ModuleList()

        # build shared layers
        if self.num_shared > 0:
            shared_layers = nn.ModuleList()

            for i in range(self.num_shared):

                if i == 0:
                    shared_layers.append(
                        nn.Linear(self.reprs_dims, self.reprs_dims * 2, bias=False)
                    )
                else:
                    shared_layers.append(
                        nn.Linear(self.reprs_dims * 2, self.reprs_dims * 2, bias=False) # TODO check
                    )
        else:
            shared_layers = None 

        # build step-specified layers
        for step in range(self.num_steps):
            self.feats_transformers.append(
                FeatureTransformer(self.reprs_dims, self.reprs_dims, shared_layers, self.num_indep, self.virtual_batch_size, self.momentum)
            )

            self.dense_layers = nn.Linear(self.reprs_dims, self.input_dims, bias=False)

    def forward(self, x):
        """
        params: x: outputs form TabNetEncoder.
        """
        outout = 0

        for step in range(self.num_steps):
            o = self.feats_transformers[step](x[step])
            o = self.dense_layers[step]


class _LabelEncoderContainer:
    def __init__(self):
        self._label_encoders = None 

    def add_encoder(self, encoder):
        if self._label_encoders is None:
            self._label_encoders = []
        self._label_encoders.append(encoder)

    def check_data(self):
        raise NotImplementedError

    def __call__(self, x, index):
        return self._label_encoders[index](x)


class EmbeddingEncoder(nn.Module):
    """
    Implementation of Embedding Encoder for simple data pre-processing of raw input features. 

    Note: `Unseen class in inference phase` issue
    """
    def __init__(self, cate_indices, cate_embed_dims, label_encoder_path=None):
        super(EmbeddingEncoder, self).__init__():

        if isinstance()
        self.cate_indices = cate_indices 
        self.cate_embed_dims = cate_embed_dims
        self.label_encoder_path = label_encoder_path

        if label_encoder_path is not None:
            self._load_label_encoder(label_encoder_path)
        else:
            self._fit_label_encoder()

        self._build()

    def _load_label_encoder(self, path):
        pass 

    def _build(self):
        



            
