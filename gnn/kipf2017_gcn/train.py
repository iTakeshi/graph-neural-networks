import chainer
from chainer.training import extensions
import numpy as np
import scipy

from gnn.common.util import find_data_local, sparse_scipy_to_chainer
from gnn.kipf2017_gcn.model import GCN, GCNTrainer
from gnn.kipf2017_gcn.util import load_data, normalize_adj


def train(dataset_str, exp_id, device=-1, max_epoch=200):
    if device > 0:
        chainer.backends.cuda.get_device_from_id(device).use()

    adj, features, y_train, y_val, y_test, mask_train, mask_val, mask_test = load_data(dataset_str)
    adj = normalize_adj(adj + scipy.sparse.eye(adj.shape[0]))
    if device > 0:
        adj = sparse_scipy_to_chainer(adj, chainer.backends.cuda.cupy)
    else:
        adj = sparse_scipy_to_chainer(adj, np)
    features /= features.sum(axis=1).reshape((-1, 1))

    model = GCNTrainer(GCN(adj, features.shape[1], 16, np.max(y_train) + 1, 0.5), 5e-4, mask_train, mask_val)

    if device > 0:
        model.to_gpu()

    iter_train = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(features, y_train), batch_size=features.shape[0], shuffle=False)
    iter_val   = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(features, y_val  ), batch_size=features.shape[0], shuffle=False, repeat=False)

    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(model)
    updater = chainer.training.StandardUpdater(iter_train, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (max_epoch, "epoch"), out=(find_data_local(__file__) / "result" / exp_id / "snapshot"))
    trainer.extend(extensions.Evaluator(iter_val, model, device=device))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss',
        'validation/main/loss',
        'main/accuracy',
        'validation/main/accuracy',
        'elapsed_time',
    ]))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    chainer.serializers.save_npz(find_data_local(__file__) / "result" / exp_id / "model", model.gcn)


if __name__ == "__main__":
    train("cora", "test00", device=-1)
