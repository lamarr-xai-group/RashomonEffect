import torch
from torch import nn

def make_ff(shapes_layers, actfun_out=None):
    # shapes layers: output size layer is input size of next ..
    layers = [nn.Linear(s_in, s_out) for (s_in, s_out) in zip(shapes_layers[:-1], shapes_layers[1:])]
    actfun = nn.ReLU
    architecture = []
    for layer in layers:
        architecture.append(layer)
        architecture.append(actfun())
    architecture = architecture[:-1] # delete last actfun
    if actfun_out is not None:
        architecture.append(actfun())
    sequential = nn.Sequential(*architecture)
    return SimpleNet(sequential=sequential)


class SimpleNet(nn.Module):
    def __init__(self, sequential):
        super(SimpleNet, self).__init__()
        self.net = sequential
        self.LossFun = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.net(x)
        return out

    def predict_batch(self, x):
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

    def predict_batch_softmax(self, x):
        pred = self.forward(x)
        sm_pred = torch.softmax(pred, dim=1)
        return sm_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.net(x)
        # y_pred = torch.argmax(y_pred, dim=1)
        loss = self.LossFun(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.net(x)
        # y_pred = torch.argmax(y_pred, dim=1)
        loss = self.LossFun(y_pred, y)
        self.log('val_loss', loss)


class TorchEnsembleWrapper(nn.Module):

    def __init__(self, models, out_dim, forwards=None):
        super(TorchEnsembleWrapper, self).__init__()
        self.models = models
        self.out_dim = out_dim
        self.forwards = forwards if forwards is not None else models

        self.eval()

    def forward(self, x):
        outputs = torch.zeros((len(self.forwards), x.shape[0],  self.out_dim))
        # forward full batch through each model
        for i, f in enumerate(self.forwards):
            outputs[i] = f(x)
        # re-order, s.t. predictions.shape = (len(x), len(forwards), out_dim)
        predictions = torch.swapaxes(outputs, 0, 1)
        return predictions

    def predict(self, x):
        '''Possible ties resolved by whatever torch.argmax chooses first'''
        outputs = self.forward(x)  # shape: batch, n_models, n_classes
        votes = torch.argmax(outputs, dim=-1)  # shape: batch, n_models
        counts = [torch.bincount(p) for p in votes]  # bincount inserts 0 for missing values
        predictions = [torch.argmax(count) for count in counts]  # get class label with highest count
        predictions = torch.tensor(predictions)
        return predictions

class TorchAdditiveEnsemble(nn.Module):

    def __init__(self, models):
        super(TorchAdditiveEnsemble, self).__init__()
        self.models = models
        self.eval()

    def forward(self, x):
        output = self.models[0](x)

        for model in self.models[1:]:
            output += model(x)

        # TODO: Softmax?
        return output

    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, axis=-1)



class BiLSTMClassif(nn.Module):

    def __init__(self, nr_classes, embed_dim, hid_size, vocab_size):
        super(BiLSTMClassif, self).__init__()
        # sparse limits available optimizers to SGD, SparseAdam and Adagrad
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hid_size,
            # dropout=False,
            bidirectional=True,
            batch_first=True,
        )
        self.fc_out = nn.Linear(2*hid_size, nr_classes)
        # self.return_type = -1

    def get_embeddings_variance(self):
        return torch.var(self.embedding.weight).item()

    def embed_sequences(self, seqs):
        with torch.no_grad():
            embedded = self.embedding(seqs)
        return embedded

    def forward(self, seqs): #, offsets):
        embedded = self.embedding(seqs) #, offsets)
        return self._forward_embedded(embedded)

    def _forward_embedded(self, embedded_seq):
        lstm_out, (_, _) = self.bilstm(embedded_seq)
        h_T = lstm_out[:, -1]
        y = self.fc_out(h_T)
        return y

    def _forward_embedded_softmax(self, embedded_seq):
        x = self._forward_embedded(embedded_seq)
        y = torch.softmax(x, dim=1)
        return y

    def forward_softmax(self, seqs):
        embedded = self.embedding(seqs) #, offsets)
        return self._forward_embedded_softmax(embedded)



if __name__=='__main__':
    from datasets import *
    from tqdm import tqdm
    from time import time
    from util import create_checkpoint

    def validate(model, data): # on single batch
        start = time()
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            batch_acc = []
            x, y = next(iter(data))
            x = x.to(device)
            _y = model(x)
            _y = _y.argmax(-1).to('cpu')
            batch_acc.append((_y==y).to(torch.float32).mean())

        model.train()
        end = time()
        print(f"took {end-start:.3f}s")
        return torch.Tensor(batch_acc).mean().item()

    def train_loop(model, optim, loss_fn, tr_data, te_data, n_batches=1000, device='cuda', pth='./', modelname='model'):
        print(device)
        model.to(device)
        acc_val = []
        models = []
        losses = [0.]
        print("process testset")
        acc_val.append(validate(model, te_data))
        print(acc_val[-1])
        batchnum = 0
        accs = []
        while batchnum < n_batches:
            for i, (text, labels) in enumerate(tr_data, 1):
                # if i % 10 == 0:
                    # create_checkpoint(f"{pth}{modelname}_{epoch}-{i}.torch", model, optim, epoch, None)
                acc = validate(model, te_data); accs.append(acc)
                print(f"test acc @ {batchnum}: {acc}")
                # print(f"mean loss since last check {sum(losses)/len(losses)}")
                # models.append(copy.deepcopy(model).to('cpu'))
                text = text.to(device)
                labels = labels.to(device)
                out = model(text)
                loss = loss_fn(out, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
                batchnum += 1
            print(f'keeping {len(models)} models in memory')
        print("accuracies over test set")
        print(acc_val)
        return accs


    print("TEST MODELS")
    from torch.optim import Adam

    # train, test, n_dim, n_classes = get_emnist(42)
    # cnn_emnist = make_fmnist_small(n_classes=n_classes)
    # path = './data/'
    # modelname= 'cnn_emnist'
    # optim = Adam(cnn_emnist.parameters())
    # lfun = torch.nn.CrossEntropyLoss()
    # train_loop(cnn_emnist, optim, lfun, train, test, device='cuda', pth=path, modelname=modelname, n_epoch=15)
    #
    # exit()
    # train, test, n_dim, n_classes = get_ionosphere(random_state=42, batch_size=4)
    # train, test, n_dim, n_classes = get_iris(random_state=42, batch_size=8)
    # train, test, n_dim, n_classes = get_covtype(random_state=42, batch_size=1000)
    # train, test, n_dim, n_classes = get_classification(batch_sizes=(8, 300))
    # train, test, n_dim, n_classes = get_waveform(batch_size=8)
    # train, test, n_dim, n_classes = get_breast_cancer(42)
    n_loop = 50
    accs = []

    for i in range(n_loop):
        train, test, n_dim, n_classes = get_classification(random_state=123456, batch_sizes=(32,))
        ff_cov = make_ff([n_dim,32, 32, n_classes])
        print(n_dim, n_classes)
        path = './data/'
        modelname= 'ff_wf'
        optim = Adam(ff_cov.parameters())
        lfun = torch.nn.CrossEntropyLoss()
        accs.append(train_loop(ff_cov, optim, lfun, train, test, device='cpu', pth=path, modelname=modelname, n_batches=800))
    import matplotlib.pyplot as plt
    for acc in accs:
        plt.plot(acc)
    plt.show()

    # data, testset, size_vocab, n_classes = get_agnews(random_state=42, batch_sizes=(64, 200))
    #
    # nlp_model = BiLSTMClassif(nr_classes=n_classes, embed_dim=128, hid_size=256, vocab_size=size_vocab)
    # # nlp_model = nn.DataParallel(nlp_model)
    #
    # optim = Adam(nlp_model.parameters())
    # lfun = torch.nn.CrossEntropyLoss()
    # train_loop(nlp_model, optim, lfun, data, testset)