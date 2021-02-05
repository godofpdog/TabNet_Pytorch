""" Test code for this repo. """


class _Test:
    def __init__(self):
        pass 
    
    @classmethod
    def test_gbn(cls):
        import torch 
        from src.core.model import GhostBatchNorm

        x = torch.randn(256, 100)
        gbn = GhostBatchNorm(100, 64)

        print(gbn(x).size())

    @classmethod
    def test_tab_encoder(cls):
        import torch 
        from src.core.model import TabNetEncoder
        
        input_dims_ = 64
        batch_size_ = 256
        x = torch.randn(batch_size_, input_dims_)

        tab_encoder = TabNetEncoder(input_dims_)
        outputs, mask_loss = tab_encoder(x)

        print(tab_encoder)
        print(mask_loss)

        print('========== test forward ==========')

        for output in outputs:
            print(output.size())

        print('========== test explain ==========')

        m_explain, masks = tab_encoder.explain(x)

        print(m_explain)
        print(masks.keys())
        print(masks)
        

    @classmethod
    def test_tab_head(cls):
        import torch 
        from src.core.model import TabNetHead

        reprs_dims_ = 8
        output_dims_ = [1, 3, 5]
        x = torch.randn(256, reprs_dims_)

        tabhead = TabNetHead(reprs_dims_, output_dims_)
        
        print(tabhead)
        res = tabhead(x)

        for r in res:
            print(r.size())

    @classmethod
    def test_embedding_encoder(cls):
        import torch
        import numpy as np
        from sklearn.datasets import load_boston
        from src.core.model import EmbeddingEncoder

        X, _ = load_boston(return_X_y=True)
        cates_feats = np.random.choice((0, 1, 2), size=X.shape[0])

        print('X : ', X.shape)
        print('cates_feats : ', cates_feats.shape)
        print(cates_feats)

        cate_X = np.hstack((X, cates_feats.reshape(-1, 1)))
        print(cate_X.shape)

        input_dims = cate_X.shape[1]
        cate_indices = [13]
        cate_dims = [3]
        embed_dims = 1

        embedding_encoder = EmbeddingEncoder(
            input_dims, cate_indices, cate_dims, embed_dims
        )

        embeds = embedding_encoder(torch.from_numpy(cate_X))

        print(embeds.size())

    # @classmethod
    # def test_inference_model(cls):
    #     import torch 
    #     import numpy as np
    #     from sklearn.datasets import load_boston
    #     from src.core.model import InferenceModel

    #     X, _ = load_boston(return_X_y=True)
    #     cates_feats = np.random.choice((0, 1, 2), size=X.shape[0])
    #     cate_X = np.hstack((X, cates_feats.reshape(-1, 1)))

    #     print('X : ', X.shape)
    #     print('cates_feats : ', cates_feats.shape)
    #     print('cate_X : ', cate_X.shape)

    #     input_dims = cate_X.shape[1]
    #     output_dims = [8, 32]
    #     cate_indices = [13]
    #     cate_dims = [3]
    #     embed_dims = 1

    #     infer_model = InferenceModel(
    #         input_dims, output_dims, cate_indices, cate_dims, embed_dims
    #     )

    #     print(infer_model)

    #     print('========== test forward ==========')
    #     outputs, m_loss = infer_model(torch.from_numpy(cate_X))

    #     for o in outputs:
    #         print(o.size())

    #     print(m_loss)

    #     print('========== test explain ==========')
    #     m_explain, masks = infer_model.explain(torch.from_numpy(cate_X))

    #     print('m_explain : ', m_explain.size())
    #     print('masks : ', masks)

    @classmethod
    def test_dataset(cls):
        from sklearn.datasets import load_boston
        from sklearn.model_selection import train_test_split
        from src.core.data import TabularDataset

        X, y = load_boston(return_X_y=True)

        print(X.shape, y.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        train_dataset = TabularDataset(X_train, y_train)

        print(X_train.shape)

        print(len(train_dataset))

        data = train_dataset[0]
        x, y = data 

        print(x.shape, y.shape)

    @classmethod
    def test_data_loader(cls):
        from sklearn.datasets import load_boston
        from src.core.data import create_data_loader

        X, y = load_boston(return_X_y=True)

        print(X.shape, y.shape)

        data_loader = create_data_loader(
            X, y, batch_size=4, num_workers=1, pin_memory=False
        )

        for i, data in enumerate(data_loader):
            x, y = data 
            print(i, x.shape, y.shape)

    @classmethod
    def test_init_fc_layer(cls):
        import torch 
        from src.core.model_builder import _init_weights

        model = torch.nn.Linear(32, 64)

        for m in model.modules():
            print(m.weight.size())

        _init_weights(model)

    @classmethod
    def test_init_glu(cls):
        import torch 
        from src.core.model import GLUBlock
        from src.core.model_builder import _init_weights

        glu = GLUBlock(32, 64, None, 128, 1.3)

        for m in glu.modules():
            print('============')
            print(m)
            
            if isinstance(m, torch.nn.BatchNorm1d):
                print('bn')
                print(m.weight.size())
                _init_weights(m, True)

            if isinstance(m, torch.nn.Linear):
                print('fc')
                print(m.weight.size())
                _init_weights(m, True)

    @classmethod
    def test_tabnet_encoder_builder(cls):
        import torch 
        from src.core.model_builder import _TabNetEncoderBuilder
        from src.core.model_builder import _init_weights

        builder = _TabNetEncoderBuilder()
        model = builder.build(
            input_dims=128
        )

        print(model)

    @classmethod
    def test_build_model_func(cls):
        import torch 
        import numpy as np
        from sklearn.datasets import load_boston
        from src.core.model_builder import build_model

        # prepare data
        X, _ = load_boston(return_X_y=True)
        cates_feats = np.random.choice((0, 1, 2), size=X.shape[0])
        cate_X = np.hstack((X, cates_feats.reshape(-1, 1)))

        print('X : ', X.shape)
        print('cates_feats : ', cates_feats.shape)
        print('cate_X : ', cate_X.shape)

        # model configs
        model_configs = {
            'input_dims': cate_X.shape[1], 'output_dims': [8, 12], 
            'cate_indices': [13], 'cate_dims': [3], 'embed_dims': 8
        }

        model = build_model(
            model_type='inference_model', weights_path=None, is_cuda=False, **model_configs
        )

        print(model)

        outputs, m_loss = model(torch.from_numpy(cate_X))

        print('m_loss : ', m_loss)

        for o in outputs:
            print(o.size())


class TabNetDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        n_d=8,
        n_steps=3,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.
        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        """
        super(TabNetDecoder, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        self.feat_transformers = torch.nn.ModuleList()
        self.reconstruction_layers = torch.nn.ModuleList()

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))
                else:
                    shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))

        else:
            shared_feat_transform = None

        for step in range(n_steps):
            transformer = FeatTransformer(
                n_d,
                n_d,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)
            reconstruction_layer = Linear(n_d, self.input_dim, bias=False)
            initialize_non_glu(reconstruction_layer, n_d, self.input_dim)
            self.reconstruction_layers.append(reconstruction_layer)

    def forward(self, steps_output):
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feat_transformers[step_nb](step_output)
            x = self.reconstruction_layers[step_nb](step_output)
            res = torch.add(res, x)
        return res


class TabNetPretraining(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        pretraining_ratio=0.2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        super(TabNetPretraining, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )
        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=n_independent,
            n_shared=n_shared,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    def forward(self, x):
        """
        Returns: res, embedded_x, obf_vars
            res : output of reconstruction
            embedded_x : embedded input
            obf_vars : which variable where obfuscated
        """
        embedded_x = self.embedder(x)
        if self.training:
            masked_x, obf_vars = self.masker(embedded_x)
            # set prior of encoder with obf_mask
            prior = 1 - obf_vars
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obf_vars
        else:
            steps_out, _ = self.encoder(embedded_x)
            res = self.decoder(steps_out)
            return res, embedded_x, torch.ones(embedded_x.shape).to(x.device)

    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)


class RandomObfuscator(torch.nn.Module):
    """
    Create and applies obfuscation masks
    """

    def __init__(self, pretraining_ratio):
        """
        This create random obfuscation for self suppervised pretraining
        Parameters
        ----------
        pretraining_ratio : float
            Ratio of feature to randomly discard for reconstruction
        """
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio

    def forward(self, x):
        """
        Generate random obfuscation mask.
        Returns
        -------
        masked input and obfuscated variables.
        """
        obfuscated_vars = torch.bernoulli(
            self.pretraining_ratio * torch.ones(x.shape)
        ).to(x.device)
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_vars


if __name__ == '__main__':
    _Test.test_build_model_func()
