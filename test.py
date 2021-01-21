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


if __name__ == '__main__':
    _Test.test_build_model_func()
