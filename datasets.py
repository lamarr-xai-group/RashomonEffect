import os
import torch
import torchvision
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from torchdata.datapipes.iter import Shuffler, Sampler
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import DBpedia, AG_NEWS, YelpReviewFull

from sklearn.datasets import \
    make_classification, fetch_covtype, load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------------------------------------------

DATA_ROOT = './datasets'

# ----------------------------------------------------------------------------------------------------------------------

class NumpyRandomSeed(object):
    """
    Class to be used when opening a with clause. On enter sets the random seed for numpy based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        np.random.set_state(self.prev_random_state)


class TorchRandomSeed(object):
    """
    Class to be used when opening a with clause. On enter sets the random seed for torch based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = torch.get_rng_state()
        torch.set_rng_state(torch.manual_seed(self.seed).get_state())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.set_rng_state(self.prev_random_state)

# ----------------------------------------------------------------------------------------------------------------------


def _train_test_split(X, y, train_size=0.5, random_state=42):
    """
    Randomly splits data into train and test sets, keeping 100*train_size percent as training data
    :param X: array with training data, first dim is sample
    :param y: array of labels
    :param train_size: float, (0, 1], how much of the data is used for the training set
    :param random_state: Seed/ Random State used for shuffling
    :return: X_train, X_test, y_train, y_test
    """
    with NumpyRandomSeed(random_state):
        nr_samples = len(X)
        split = int(train_size * nr_samples)
        assert split <= nr_samples
        idxs = np.arange(nr_samples)
        np.random.shuffle(idxs)
        # x train, x val, y train, y val
    return X[idxs[:split]], X[idxs[split:]], y[idxs[:split]], y[idxs[split:]]


def _return_dataset(data: np.ndarray, target: np.ndarray, batch_size, train_size: float, as_torch, random_state):
    """
    train-test-split sklearn datasets and return as DataLoader or as numpy arrays
    :param data: data in a numpy array
    :param target: labels in numpy array
    :param batch_size: size of the batches the DataLoader returns
    :param train_size: float in (0, 1], share of data used for training data
    :param as_torch: bool, if True return DataLoader, else return
    :param random_state:
    :return: Dataloder/ tuple(array, array) training set, Dataloder/ tuple(array, array) test set, int dimensionality of data, int number of classes
    """
    X_tr, X_te, Y_tr, Y_te = _train_test_split(data, target, train_size=train_size, random_state=random_state)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    n_dim = X_tr.shape[1]
    n_classes = len(np.unique(Y_tr))

    assert np.all(np.unique(Y_tr) == np.unique(Y_te))

    if not as_torch:
        return (X_tr, Y_tr), (X_te, Y_te), n_dim, n_classes
    else:
        # gen_train = torch.Generator('cpu'); gen_train.manual_seed(random_state)
        # gen_test = torch.Generator('cpu'); gen_test.manual_seed(random_state)
        gen_train, gen_test = None, None
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).long()),
                                  shuffle=True, batch_size=batch_size[0], generator=gen_train)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).long()),
                                 shuffle=False, batch_size=batch_size[-1], generator=gen_test)
        return train_loader, test_loader, n_dim, n_classes


def _make_waveform(n=300, random_state=None, batch_sizes=None):
    """ create waveform data """
    if random_state is None:
        npr = np.random.RandomState(1234)
    elif isinstance(random_state, int):
        npr = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        npr = random_state
    else:
        raise Exception('Unknown input type of random state:', type(random_state))

    y = npr.randint(3, size=n)
    u = npr.uniform(size=n)
    x = npr.normal(size=(n, 21))
    m = np.arange(1, 22)
    h1 = np.maximum(0, 6 - np.abs(m - 7))
    h2 = np.maximum(0, 6 - np.abs(m - 15))
    h3 = np.maximum(0, 6 - np.abs(m - 11))
    x[y == 0] += np.outer(u[y == 0], h1) + np.outer(1 - u[y == 0], h2)
    x[y == 1] += np.outer(u[y == 1], h1) + np.outer(1 - u[y == 1], h3)
    x[y == 2] += np.outer(u[y == 2], h2) + np.outer(1 - u[y == 2], h3)

    return x, y


def _get_vocab(name='AG_NEWS', train_iter=None):
    vocab_path = f"{DATA_ROOT}/{name}/vocab.torch"
    print(f"looking for vocab in {vocab_path}")
    try:
        vocab = torch.load(vocab_path)
    except FileNotFoundError:
        print("Vocab not found, building vocab ...")
        tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        torch.save(vocab, vocab_path)
        print(f"... done, saved vocab of {len(vocab)} words in {vocab_path}")
    return vocab


def _build_collate_fn(vocab, label_pipeline):
    """
    given a text dataset, returns preprocessing function as required for DataLoader
    :param train_iter: iterator over text dataset
    :return:
    """
    tokenizer = get_tokenizer('basic_english')
    padding_val = vocab['<pad>']

    # label_pipeline = lambda x: int(x) - 1
    # text_pipeline = lambda x: vocab(tokenizer(x))
    def text_pipeline(input):
        return vocab(tokenizer(input))

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            # offsets.append(processed_text.size(0))
        labels = torch.tensor(label_list, dtype=torch.int64)
        # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        # text_list = torch.cat(text_list)
        text = pad_sequence(text_list, batch_first=True, padding_value=padding_val)
        return text, labels #, offsets

    return collate_batch, len(vocab)

# ----------------------------------------------------------------------------------------------------------------------


"""
    Functions that return a torch DataLoader that wraps the respective dataset.
    Datasets loaded via sklearn can also be returned as numpy arrays.
"""

# --------------- SKLEARN


def get_covtype(random_state, batch_sizes=(64, 1050), train_size=0.8, as_torch=True):
    # dim = 54, classes = 7
    covertype_bunch = fetch_covtype(data_home=DATA_ROOT)
    data, target = covertype_bunch.data, covertype_bunch.target
    target -= 1  # covtype targets start with 1 not 0
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)


def get_iris(random_state, batch_sizes=(64,450), train_size=0.8, as_torch=True):
    # dim = 4, classes = 3
    data, target = load_iris(return_X_y=True)
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)


def get_wine(random_state, batch_sizes=(64,450), train_size=0.8, as_torch=True):
    # dim = 13, classes = 3
    data, target = load_wine(return_X_y=True)
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)


def  get_breast_cancer(random_state, batch_sizes=(64,300), train_size=0.8, as_torch=True):
    # dim = 30, classes =2
    data, target = load_breast_cancer(return_X_y=True)
    return _return_dataset(data, target, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)


def get_classification(nr_samples=10000, random_state_data=6810267, random_state_split=185619,
                       batch_sizes=(64,450), train_size=0.8, as_torch=True, random_state=None):
    X, y = make_classification(n_samples=nr_samples, n_features=20, n_informative=13, n_redundant=0, n_classes=3,
                               # flip_y=0.03, class_sep=0.1,
                               n_repeated=0, shuffle=False, random_state=random_state_data)
    return _return_dataset(X, y, batch_sizes, train_size, as_torch=as_torch, random_state=random_state_split)


# --------------- TORCH

# ---------------------- VISION

def get_fmnist(random_state, batch_sizes=(64,512), root=DATA_ROOT):
    """load Fashion MNIST from torchvision"""
    # dim = 28x28, classes = 10
    with TorchRandomSeed(random_state):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        data = torchvision.datasets.FashionMNIST

        train = data(root, train=True, download=True, transform=transform)
        test = data(root, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_sizes[0], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sizes[-1], shuffle=False)

    return train_loader, test_loader, (28,28), 10

def get_emnist(random_state, batch_sizes=(64,512), root=DATA_ROOT):
    """load extended-MNIST from torchvision"""
    with TorchRandomSeed(random_state):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        data = torchvision.datasets.EMNIST

        train = data(root, split='balanced', train=True, download=True, transform=transform)
        test = data(root, split='balanced', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_sizes[0], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sizes[-1], shuffle=False)

    return train_loader, test_loader, (28, 28), 47


# ---------------------- NLP

def get_agnews(random_state, batch_sizes=(64, 200), root=DATA_ROOT):
    """load AGNews dataset from torchtext"""
    # -, classes = 4
    def label_pipeline(input):
        return int(input) - 1

    with TorchRandomSeed(random_state):
        # base vocab in collate function on training data
        gen_train = torch.Generator(); gen_train.manual_seed(random_state)
        gen_test = torch.Generator(); gen_test.manual_seed(random_state)

        train_iter = AG_NEWS(root, split='train')
        train_iter = to_map_style_dataset(train_iter)

        test_iter = AG_NEWS(root, split='test')
        test_iter = to_map_style_dataset(test_iter)

        vocab = _get_vocab('AG_NEWS', train_iter)
        collate_batch, size_vocab = _build_collate_fn(vocab, label_pipeline)

        train_loader = DataLoader(train_iter, batch_size=batch_sizes[0],
                                  shuffle=True, collate_fn=collate_batch, generator=gen_train)
        test_loader = DataLoader(test_iter, batch_size=batch_sizes[-1], shuffle=True, collate_fn=collate_batch,
                                 generator=gen_test)

    return train_loader, test_loader, size_vocab, 4


def get_dbpedia(random_state, batch_sizes=(64, 128), root=DATA_ROOT):
    """load DBPedia dataset from torchtext"""
    # -, classes = 14
    if not os.path.exists(DATA_ROOT+'/DBpedia/dbpedia_csv.tar.gz'):
        print(f"DBpedia dataset not found. In case of error, download manually from\n"
              f"https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k\n"
              f"and place dbpedia_csv.tar.gz in root/DBpedia")

    def label_pipeline(input):
        return int(input)-1

    with TorchRandomSeed(random_state):
        # base vocab in collate function on training data
        train_iter = DBpedia(root, split='train')
        train_iter = to_map_style_dataset(train_iter)
        test_iter = DBpedia(root, split='test')
        test_iter = to_map_style_dataset(test_iter)

        vocab = _get_vocab('DBpedia', train_iter)
        collate_batch, size_vocab = _build_collate_fn(vocab, label_pipeline,)

        train_loader = DataLoader(train_iter, batch_size=560000, shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(test_iter, batch_size=batch_sizes[-1], shuffle=True, collate_fn=collate_batch)

    return train_loader, test_loader, size_vocab, 14

def get_yelpreviewfull(random_state, batch_sizes=(64, 128), root=DATA_ROOT):
    """load YelpReviewFull datasedt from torchtext"""
    # -, classes = 5
    if not os.path.exists(DATA_ROOT+'/YelpReviewFull/yelp_review_full_csv.tar.gz'):
        print(f"YelpReviewFull dataset not found. In case of error, download manually from\n"
              f"https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg\n"
              f"and place yelp_review_polarity_csv.tar.gz in project_root/YelpReviewFull")

    def label_pipeline(input):  # think this doesnt work?
        return int(input)-1

    with TorchRandomSeed(random_state):
        # base vocab in collate function on training data
        train_iter = YelpReviewFull(root, split='train')
        train_iter = to_map_style_dataset(train_iter)

        test_iter = YelpReviewFull(root, split='test')
        test_iter = to_map_style_dataset(test_iter)

        vocab = _get_vocab('YelpReviewFull', train_iter)
        collate_batch, size_vocab = _build_collate_fn(vocab, label_pipeline)

        train_loader = DataLoader(train_iter, batch_size=batch_sizes[0], shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(test_iter, batch_size=batch_sizes[-1], shuffle=True, collate_fn=collate_batch)

    return train_loader, test_loader, size_vocab, 5


# --------------- MISC


def get_waveform(nr_samples=10000, random_state_data=10419187, random_state_split=185619,
                       batch_sizes=(64, 450), train_size=0.8, as_torch=True, random_state=None):
    X, y = _make_waveform(nr_samples, random_state_data)
    return _return_dataset(X, y, batch_sizes, train_size, as_torch=as_torch, random_state=random_state_split)


def get_ionosphere(random_state, batch_sizes=(64, 300), train_size=0.8, as_torch=True, root=DATA_ROOT):
    """load ionosphere dataset from DATA_ROOT subfolder"""
    pth = root+'/IONOSPHERE/ionosphere.data'

    X = np.genfromtxt(pth, delimiter=',', usecols=np.arange(34))
    _y = np.genfromtxt(pth, delimiter=',', dtype=str, usecols=[34])
    y = np.array([0 if l == 'g' else 1 for l in _y])
    return _return_dataset(X, y, batch_sizes, train_size, as_torch=as_torch, random_state=random_state)

def get_beans(random_state, batch_sizes=(32, 1050), train_size=0.8, as_torch=True, root=DATA_ROOT):
    """load DryBeansDataset dataset from DATA_ROOT subfolder"""
    # 16, 7
    pth = root+'/DryBeansDataset/Dry_Beans_Dataset.csv'
    try:
        X = np.genfromtxt(pth, delimiter=';', dtype=float, usecols=np.arange(16), skip_header=1)
        _y = np.genfromtxt(pth, delimiter=';', dtype=str, usecols=[16], skip_header=1)
    except OSError as o:
        print(o)
        print(f"Get Dry Beans Dataset at https://archive-beta.ics.uci.edu/ml/datasets/dry+bean+dataset"
              f"Convert Excel to ;-separated CSV, replace comma in numbers by dot"
              f"Place in {pth}")
        exit()
    classmap = {c: n for n, c in enumerate(np.unique(_y))}
    y = np.array([classmap[c] for c in _y])

    (X_tr, Y_tr), (X_te, Y_te), n_dim, n_classes = _return_dataset(X, y, batch_sizes, train_size,
                                                                  random_state=random_state, as_torch=False)
    # normalize columns to unit variance
    mu = np.mean(X, 0)
    std = np.std(X, 0)
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    if as_torch:
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(Y_tr).long()),
                                shuffle=True, batch_size=batch_sizes[0])
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(Y_te).long()),
                                shuffle=False, batch_size=batch_sizes[-1])

        return train_loader, test_loader, n_dim, n_classes
    else:
        return (X_tr, Y_tr), (X_te, Y_te), n_dim, n_classes






# --------------- HELPER ---------------

__info_dim_classes = {
    'covtype': (54, 7),
    'iris': (4, 3),
    'wine': (13, 3),
    'breastcancer': (30, 2),
    'classification': (20, 3),
    'beans': (16, 7),

    'fmnist': ((1, 28, 28), 10),
    'emnist': ((1, 28, 28), 47),

    'agnews': (95812, 4),
    'dbpedia': (802999, 14),
    'yelpreviewfull': (519820, 5),

    'waveform': (21, 3),
    'ionosphere': (34, 2)
}

def _get_dim_classes(dataset : str):
    return __info_dim_classes[dataset]

dataset_callables = {
    'covtype': get_covtype,
    'iris': get_iris,
    'wine': get_wine,
    'breastcancer': get_breast_cancer,
    'classification': get_classification,
    'beans': get_beans,

    'fmnist': get_fmnist,
    'emnist': get_emnist,

    'agnews': get_agnews,
    'dbpedia': get_dbpedia,
    'yelpreviewfull': get_yelpreviewfull,

    'waveform': get_waveform,
    'ionosphere': get_ionosphere
}

def _get_dataset_callable(dataset: str):
    return dataset_callables[dataset]


nlp_tasks = ['agnews', 'dbpedia', 'yelpreviewfull']
cv_tasks = ['fmnist', 'emnist']
tabular_tasks = [k for k in dataset_callables.keys() if k not in nlp_tasks and k not in cv_tasks]  # :)


# ----------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':
    from tqdm import tqdm
    from copy import deepcopy

    def assess_randomization(loader1, loader2, D=3, seed=199):
        def get_some_batches(loader):
            X, Y = [], []

            n_loops = 3
            n_batches = 3
            # sample n_loops times n_batches from both train and test
            for e in range(n_loops):
                X.append([]); Y.append([])
                for i, (x1, y1) in enumerate(loader):
                    if i > n_batches:
                        break
                    X[-1].append(x1); Y[-1].append(y1)
            return X, Y

        x1, y1 = get_some_batches(loader1)
        x2, y2 = get_some_batches(loader2)
        with TorchRandomSeed(seed):
            x1r, y1r = get_some_batches(loader1)
        with TorchRandomSeed(seed + 1):
            x2r, y2r = get_some_batches(loader2)

        # assert that l1 == l2 and l1_rand != l2_rand

        # data = [[tr_data[d][0] for d in range(D)], [te_data[d][1] for d in range(D)]]
        print('done')
        pass

    def check_data(train, test, dim, n_classes):
        print(f"{dim}, {n_classes}")
        print(f"sizes train: {len(train)} test {len(test)}")
        print(f"dim {dim}, n_classes {n_classes}")
        for (x_tr, y_tr), (x_te, y_te) in zip(train, test):
            print("shapes train data - labels: ", x_tr.shape, y_tr.shape)
            print("shapes test data - labels: ", x_te.shape, y_te.shape)
            print(x_tr)
            print(x_te)
            break

    print("############################")
    print("### TESTING ALL DATAESTS ###")
    print("############################\n\n")

    t1, _,_,_ = get_agnews(random_state=32, batch_sizes=(1, 1))
    l = []
    for x, y in t1:
        l.append(x.shape[1])
    print(sum(l)/len(l))

    random_state = 42
    with TorchRandomSeed(random_state):
        tr1, te1, _, _ = get_agnews(123)
    with TorchRandomSeed(random_state):
        tr2, te2, _, _ = get_agnews(123)
    assess_randomization(tr1, tr2)
    tr3, te3, _, _ = get_agnews(132)
    assess_randomization(tr1, tr3)
    assess_randomization(te3, te3)
    exit()
    import sys
    sys.exit()


    # train, test, n_feat, n_classes = get_beans(random_state)
    # check_data(train, test, n_feat, n_classes)
    # train, test, size_vocab, n_classes = get_dbpedia(random_state)
    # check_data(train, test, size_vocab, n_classes)
    # train, test, size_vocab, n_classes = get_agnews(random_state)
    # check_data(train, test, size_vocab, n_classes)
    # train, test, n_feat, n_classes = get_emnist(random_state, batch_sizes=(64, 18800))
    # check_data(train, test, n_feat, n_classes)
    train, test, n_feat, n_classes = get_covtype(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### GENERATE MAKE_CLASSIFICATION")
    train, test, n_feat, n_classes = get_classification()
    check_data(train, test, n_feat, n_classes)
    print("### GENERATE WAVEFORM")
    train, test, n_feat, n_classes = get_waveform()
    check_data(train, test, n_feat, n_classes)
    print("### LOADING COVERTYPE")
    train, test, n_feat, n_classes = get_covtype(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### IONOSPHERE")
    train, test, n_feat, n_classes = get_ionosphere(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### IRIS")
    train, test, n_feat, n_classes = get_iris(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### WINE")
    train, test, n_feat, n_classes = get_wine(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### BREAST CANCER")
    train, test, n_feat, n_classes = get_breast_cancer(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### LOAD FASHION-MNIST")
    train, test, n_feat, n_classes = get_fmnist(random_state)
    check_data(train, test, n_feat, n_classes)
    print("### LOAD EXTENDED-MNIST")
    train, test, n_feat, n_classes = get_emnist(random_state, batch_sizes=(64, 18800))
    check_data(train, test, n_feat, n_classes)
    print("### LOAD YELPREVIEWSFULL")
    train, test, size_vocab, n_classes = get_yelpreviewfull(random_state)
    check_data(train, test, size_vocab, n_classes)
    print("### LOAD AG NEWS")
    train, test, size_vocab, n_classes = get_agnews(random_state)
    check_data(train, test, size_vocab, n_classes)
    print("### LOAD DBPEDIA")
    train, test, size_vocab, n_classes = get_dbpedia(random_state)
    check_data(train, test, size_vocab, n_classes)
    print("DONE")
