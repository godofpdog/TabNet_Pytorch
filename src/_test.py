import torch 
import numpy as np 

class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        """This is an embedding module for an entire set of features
        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(
            input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims)
        )

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)

        print('====sorted_idxs = ', sorted_idxs)
        print(sorted_idxs)

        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        print('*********************************')
        print(self.continuous_idx)
        print('*********************************')
        print(len(self.embeddings))

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            print('---------------------')
            print('feat_init_idx : ', feat_init_idx, 'is_continuous : ', is_continuous)
            print('embedding : ', self.embeddings[cat_feat_counter])
            if is_continuous:
                print('is conti')
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                print('is cate')
                print()
                print(x[:, feat_init_idx].max())
                print(x[:, feat_init_idx].long().size())
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings


def get_num_unique(x):
    return len(np.unique(x))


if __name__ == "__main__":
    # from sklearn.datasets import load_boston

    # X, y = load_boston(return_X_y=True)
    # print(X.shape, y.shape)

    # print(X[:, 1].shape)

    # # emb = EmbeddingGenerator(13, [3, 10, 4, 5], [10, 8, 1, 9], 32)
    # emb = EmbeddingGenerator(13, [26], [1], 32)

    # X_tensor = torch.from_numpy(X)
    # print(emb(X_tensor))


    from sklearn.preprocessing import LabelEncoder

    A = np.array(['y', 'x', 'x', 'y', 'x', 'y'])

    le = LabelEncoder()

    label_A = le.fit_transform(A)

    print(label_A)

    print(le.transform(A))

    print(le.transform(np.array([1])))

