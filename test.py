""" Test code for this repo. """

import torch 


class _Test:
    def __init__(self):
        pass 
    
    @classmethod
    def test_gbn(cls):
        from src.model import GhostBatchNorm

        x = torch.randn(256, 100)
        gbn = GhostBatchNorm(100, 64)

        print(gbn(x).size())

    @classmethod
    def test_tab_encoder(cls):
        import torch 
        from src.model import TabNetEncoder
        
        input_dims_ = 64
        batch_size_ = 256
        x = torch.randn(batch_size_, input_dims_)

        tab_encoder = TabNetEncoder(input_dims_)
        outputs, mask_loss = tab_encoder(x)

        print(tab_encoder)
        print(mask_loss)

        for output in outputs:
            print(output.size())
        

    @classmethod
    def test_tab_head(cls):
        from src.model import TabNetHead

        reprs_dims_ = 8
        output_dims_ = [1, 3, 5]
        x = torch.randn(256, reprs_dims_)

        tabhead = TabNetHead(reprs_dims_, output_dims_)
        
        print(tabhead)
        res = tabhead(x)

        for r in res:
            print(r.size())

    @classmethod
    def test_dataset(cls):
        from sklearn.datasets import load_boston
        from sklearn.model_selection import train_test_split
        from src.data import TabularDataset

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
        from src.data import create_data_loader

        X, y = load_boston(return_X_y=True)

        print(X.shape, y.shape)

        data_loader = create_data_loader(
            X, y, batch_size=4, num_workers=1, pin_memory=False
        )

        for i, data in enumerate(data_loader):
            x, y = data 
            print(i, x.shape, y.shape)


if __name__ == '__main__':
    _Test.test_tab_encoder()
