import numpy as np

###
# From 'Disagreement problem' paper http://arxiv.org/abs/2202.01602
###

def _percent_to_int(k, n):
    assert 0. < k <= 1.
    return int(np.ceil(k*n))

def _top_k_feature_agreement(a, b, k=3):
    if k < 1:
        k = _percent_to_int(k, a.shape[0])
    _top_k_a = np.argsort(np.abs(a))[:, -k:][:, ::-1]  # [:, ::-1] to get ascending order
    _top_k_b = np.argsort(np.abs(b))[:, -k:][:, ::-1]

    agreements = np.zeros(a.shape[0])

    for i, (ka, kb) in enumerate(zip(_top_k_a, _top_k_b)):
        agreements[i] = len(set(ka).intersection(set(kb))) / k

    return agreements

def _top_k_rank_agreement(a, b, k=3):
    if k < 1:
        k = _percent_to_int(k, a.shape[0])
    _top_k_a = np.argsort(np.abs(a))[:, -k:][:, ::-1]  # [:, ::-1] to get ascending order
    _top_k_b = np.argsort(np.abs(b))[:, -k:][:, ::-1]

    def rank_agreement(_a, _b, _k):
        s = 0
        for idx in range(k):
            if _a[idx] in _b and _a[idx] == _b[idx]:
                s += 1
        return s / _k

    agreements = np.zeros(a.shape[0])
    for i, (ka, kb) in enumerate(zip(_top_k_a, _top_k_b)):
        agreements[i] = rank_agreement(ka, kb, k)

    return agreements

def _top_k_sign_agreement(a, b, k=3):
    if k < 1:
        k = _percent_to_int(k, a.shape[0])
    _top_k_a = np.argsort(np.abs(a))[:, -k:][:, ::-1]  # [:, ::-1] to get ascending order
    _top_k_b = np.argsort(np.abs(b))[:, -k:][:, ::-1]

    signs = a*b

    def _sign_agreement(_ka, _kb, _s, _k):# _a, _b, _k):
        s = 0
        for idx in range(_k):
            if _ka[idx] in _kb and _s[_ka[idx]] >= 0:  # to check if top feature overlaps
                # _kbidx = np.argwhere(_kb == _ka[idx]).item()  # get idx of
                # if _a[_ka[idx]] * _b[_kbidx] > 0 or (_a[_ka[idx]] == _b[_kbidx] == 0):
                s += 1
        return s / _k

    agreements = np.zeros(a.shape[0])
    for i, (ka, kb, _s) in enumerate(zip(_top_k_a, _top_k_b, signs)):
        agreements[i] = _sign_agreement(ka, kb, _s,  k)

    return agreements

def _top_k_signed_rank_agreement(a, b, k=3):
    if k < 1:
        k = _percent_to_int(k, a.shape[0])
    _top_k_a = np.argsort(np.abs(a))[:, -k:][:, ::-1]  # [:, ::-1] to get ascending order
    _top_k_b = np.argsort(np.abs(b))[:, -k:][:, ::-1]

    # signs = a[:, _top_k_a] * b[:, _top_k_b]
    signs = np.array([a[i, _top_k_a[i]] * b[i, _top_k_b[i]] for i in range(a.shape[0])])

    def _signed_rank_agreement(_a, _b, _signs, _k):
        s = 0
        for idx in range(_k):
            # if _top_k_a[idx] in _top_k_b and signs[idx] >= 0 and _top_k_a[idx] == _top_k_b[idx]:
            # -> first statement not necessary because the third is only evaluated on top k
            if _a[idx] == _b[idx] and _signs[idx] >= 0:
                s += 1
        return s / _k

    agreements = np.zeros(a.shape[0])

    for i, (ka, kb, s) in enumerate(zip(_top_k_a, _top_k_b, signs)):
        agreements[i] = _signed_rank_agreement(ka, kb, s, k)

    return agreements


def main():


    a = np.array([[-2, -1,  3,  4], [-2, -1,  3,  4]])
    b = np.array([[ 2,  1, -3, 4 ], [-2, -1,  3,  4]])
    res = _top_k_sign_agreement(a, b)
    assert res == 1/3, res

    a = np.array([[2, 1.3, 0, 1], [-1, -2, 3, 0]])
    b = np.array([[2, 1.5, -0.5, 1], [-1.4, -1.5, 1.6, 0.]])
    res = _top_k_feature_agreement(a, b)
    _top_k_sign_agreement(a,b)
    _top_k_rank_agreement(a, b)
    _top_k_signed_rank_agreement(a, b)
    assert res == 1, res

    a = np.array([-2, 1, 0, 1])
    b = np.array([2, 1.5, -1, 1])
    assert _top_k_feature_agreement(a, b) == 1

    a = np.array([1, 2, 3, 4])
    b = np.array([4, 3, 2, 1])
    res = _top_k_feature_agreement(a, b)
    assert res == 2/3, res

    a = np.array([4, 5, 6, 1, 0])
    b = np.array([4, 0, 1, 2, 3])
    res = _top_k_feature_agreement(a, b)
    assert res == 1/3, res
   



    a = np.array([10, 9, 8, 7, 6])
    b = np.array([10, 8, 9, 7, 6])
    res = _top_k_rank_agreement(a, b)
    assert res == 1/3, res

    a = np.array([10, 8, 9, 7, 6])
    b = np.array([10, 8, 9, 7, 6])
    res = _top_k_rank_agreement(a, b)
    assert res == 1, res

    a = np.array([10, 1, 2, 7, 6])
    b = np.array([10, 7, 6, 1, 2])
    res = _top_k_rank_agreement(a, b)
    assert res == 1/3, res




    a = np.array([-2, -1,  3,  4])
    b = np.array([ 2,  1, -3, 4 ])
    res = _top_k_sign_agreement(a, b)
    assert res == 1/3, res

    a = np.array([-2, -1,  3,  4])
    b = np.array([-2, -1,  3,  4 ])
    res = _top_k_sign_agreement(a, b)
    assert res == 1, res


if __name__ == '__main__':
    main()
