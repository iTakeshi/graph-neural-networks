import pathlib

import chainer


def find_data_local(script):
    return (pathlib.Path(script).parent / "data").resolve()


def sparse_scipy_to_chainer(s, xp):
    s = s.tocoo()
    return chainer.utils.CooMatrix(
        xp.array(s.data, dtype="float32"),
        xp.array(s.row, dtype="float32"),
        xp.array(s.col, dtype="float32"),
        s.shape,
    )
