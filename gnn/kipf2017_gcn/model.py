import chainer
import chainer.functions as F


class GraphConvolution(chainer.Link):

    def __init__(self, adj, dim_i, dim_o, dropout, act):
        super().__init__()
        self.adj = adj
        self.dropout = dropout
        self.act = act
        with self.init_scope():
            self.W = chainer.Parameter(chainer.initializers.GlorotNormal(), (dim_i, dim_o))

    def forward(self, x):
        x = F.dropout(x, ratio=self.dropout)
        return self.act(F.matmul(F.sparse_matmul(self.adj, x), self.W))


class GCN(chainer.Chain):

    def __init__(self, adj, dim_i, dim_h, dim_o, dropout):
        super().__init__()
        with self.init_scope():
            self.conv1 = GraphConvolution(adj, dim_i, dim_h, dropout, F.relu)
            self.conv2 = GraphConvolution(adj, dim_h, dim_o, dropout, F.identity)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class GCNTrainer(chainer.Chain):

    def __init__(self, gcn, weight_decay, mask_train, mask_val):
        super().__init__()
        self.weight_decay = weight_decay
        self.mask_train = mask_train
        self.mask_val = mask_val
        with self.init_scope():
            self.gcn = gcn

    def forward(self, x, t):
        loss = 0
        if chainer.config.train:
            loss += F.softmax_cross_entropy(self.gcn(x)[self.mask_train], t[self.mask_train])
            chainer.reporter.report({ 'accuracy': F.accuracy(self.gcn(x)[self.mask_train], t[self.mask_train]) }, self)
        else:
            loss += F.softmax_cross_entropy(self.gcn(x)[self.mask_val], t[self.mask_val])
            chainer.reporter.report({ 'accuracy': F.accuracy(self.gcn(x)[self.mask_val], t[self.mask_val]) }, self)

        # weight decay
        # can't use optimizer.WeightDecay because GCN applys WD only for the first layer
        loss += self.weight_decay * F.sum(self.gcn.conv1.W ** 2)

        chainer.reporter.report({ "loss": loss }, self)
        return loss
