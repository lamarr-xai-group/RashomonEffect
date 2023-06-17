import torch
from torch import nn
import numpy as np
import logging

from util import TorchRandomSeed
from torch.utils.data import DataLoader, TensorDataset
from util import _get_outputs, _get_targets
from captum.attr import DeepLiftShap, DeepLift, KernelShap, Lime


def _calc_gradients(model, data, inference_fn=None, device='cpu', return_outputs=False):
    """
    calculates gradients of data wrt. labels for inference_fn on device
    :param inference_fn: callable that returns full model output for a datum
    :param data: iterable containing the data
    :param device: device to where data is loaded
    :return:
    """
    if inference_fn is None:
        inference_fn = model
    gradients = []
    if return_outputs:
        _outputs = []
    try:
        with torch.set_grad_enabled(True):
            for i, (x, y) in enumerate(data):
                x = x.to(device).requires_grad_()
                y = y.to(device)
                outputs = inference_fn(x)
                # _selected_outputs = outputs[:, y]  # since y is a tensor and not a scalar this doesn't work
                # see: e=torch.eye(10); y=torch.arange(10); torch.gather(e, 1, y.unsqueeze(1)).squeeze() return vector of just 1.'s
                _selected_outputs = torch.gather(outputs, 1, y.unsqueeze(1)).squeeze()
                grads = torch.autograd.grad(torch.unbind(_selected_outputs), x)
                gradients.append(grads[0].to('cpu').detach())
                if return_outputs:
                    _outputs.append(outputs.to('cpu').detach())

        if return_outputs:
            return torch.vstack(gradients), torch.vstack(_outputs)
        else:
            return torch.vstack(gradients)

    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            logging.warning(f"attribution._calc_gradients: CUDA out of memory. Retrying on CPU.")
            model.to('cpu')  # this is why we need the model
            # value of return_outputs doesn't matter for the return syntax
            return_value = _calc_gradients(model, data, inference_fn=inference_fn,
                                           device='cpu', return_outputs=return_outputs)
            logging.warning(f"attribution._calc_gradients: Loading model back to GPU.")
            model.to('cuda')  # putting model back
            return return_value
        else:
            raise re


def vanilla_grad(model, data, targets=None, inference_fn=None, device=None, simplified=False, pre_process_fn=None):
    return smooth_grad(inference_fn=inference_fn, model=model, data=data, targets=targets,
                       n_samples=1, std=0., random_state=0, device=device, simplified=simplified,
                       pre_process_fn=pre_process_fn)


def smooth_grad(model, data, targets=None, inference_fn=None, std=1., n_samples=50,
                random_state=42, device=None, simplified=False, noise_level=0.1, pre_process_fn=None, force_random=True):
    # expecting targets.shape=(n,)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model


    if pre_process_fn is not None:
        model = model.to('cpu')
        data = data.to('cpu')
        data = pre_process_fn(data)
        model = model.to(device)

    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)

    noise = noise_level * std
    if force_random:
        perturbation = torch.normal(mean=0., std=noise, size=(n_samples, *data.shape[1:]))
    else:
        with TorchRandomSeed(random_state):
            perturbation = torch.normal(mean=0., std=noise, size=(n_samples, *data.shape[1:]))

    if n_samples <= 1:
        perturbed_data = data
        targets_expanded = targets
        batch_size = data.shape[0]
    else:
        # each element in dim 0 is repeated n_samples times, contiguously
        perturbed_data = torch.repeat_interleave(data, repeats=n_samples, dim=0)
        targets_expanded = targets.unsqueeze(0).expand(n_samples, targets.shape[0]).reshape(-1)
        perturbation = perturbation.unsqueeze(0).expand(data.shape[0], *perturbation.shape).reshape(-1, *perturbation.shape[1:])
        batch_size = n_samples

    perturbed_data = perturbed_data + perturbation
    _data = DataLoader(TensorDataset(perturbed_data, targets_expanded),
                       shuffle=False, batch_size=batch_size)

    _prev_train_state = model.training
    model.train(True)
    _grads = _calc_gradients(model=model, data=_data, inference_fn=inference_fn, device=device)
    model.train(_prev_train_state)

    # sum everything up and out they go!
    idxs = torch.arange(0, perturbed_data.shape[0]+n_samples, n_samples)
    gradients = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        gradients.append(
            torch.mean(_grads[a:b], dim=0)
        )

    gradients = torch.stack(gradients)

    if simplified:
        gradients = torch.sum(gradients, dim=-1)

    return gradients


def integrated_gradients(model, data, targets=None, inference_fn=None, baselines=None, n_samples=100, simplified=False,
                         calc_paths=None, calc_baselines=None, fit_baseline_data=False, device=None, outputmode=None,
                         pre_process_fn=None, return_convergence_delta=False, _batch_size=256):
    """
    :param model: Classifier model
    :param data: input data, batch first
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: if none given, zero baseline is used,
        "integral approximated with left riemannian integration plus inclusion of final point",
        ie we interpolate linearly , x_0 + t(x - x_0) t = [0., ..., 1.], and include start and end points
    :param n_samples: number of steps between baseline and input -> actual steps = steps+2; only used when calc_paths is not a callable
    :param simplified: return attribution of shape (1, seqlen), accumulate attribution for each vector
            if false, returns all collected gradients of size [n_samples, n_steps, seqlen, embedding_size]
    :param calc_paths: needs to return paths between all samples as well as distance between consecutive samples
    :param calc_baselines: must return one baseline per sample
    :return:
    """

    if device is None:
        device = next(model.parameters()).device  #  this looks ugly but is apparently the way to go in vanilla pytorch

    if inference_fn is None:
        inference_fn = model

    _use_tuple = isinstance(data, tuple)

    # data = data[:2]
    # targets = targets[:2]

    if pre_process_fn is not None:
        model = model.to('cpu')
        data = data.to('cpu')
        data = pre_process_fn(data)
        model = model.to(device)

    ## put everything on cpu for now
    if _use_tuple:
        data = data[0].to('cpu')
    else:
        data = data.to('cpu')

    input = data

    # support multiple baselines per sample?
    if callable(calc_baselines):
        baselines = calc_baselines(data) # (samples, model)?
    if baselines is None:
        baselines = torch.zeros_like(input)
    elif fit_baseline_data:
        baselines = baselines.repeat(input.shape[:-1] + (1,))


    # paths.size = [n_samples, n_steps, seqlen, embedding]
    paths, scaling = None, None
    if callable(calc_paths):
        paths, scaling = calc_paths(baselines, input, n_samples)
    else:
        # fallback: straight line between baseline and input,
        p = np.linspace(baselines, input, num=n_samples)#, retstep=True)
        paths = torch.tensor(p).swapaxes(0, 1).requires_grad_()
        scaling = 1. / n_samples
        # paths = torch.tensor(np.linspace(baselines, samples, num=steps)).swapaxes(0,1).requires_grad_()
        # paths = paths[:, :-1, :]

    assert scaling is not None and paths is not None

    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)


    output_shape = inference_fn(paths[0].to(device)).shape
    # places gradients in attribution
    _prev_train_state = model.training
    model.train(True)
    attribution = torch.zeros_like(paths)
    outputs = torch.zeros(*paths.shape[:2], output_shape[1])
    for i in range(paths.shape[0]):
        _path_one_sample = paths[i]
        _target_output = targets[i].expand(paths.shape[1])
        _dataset = DataLoader(TensorDataset(_path_one_sample, _target_output), shuffle=False, batch_size=_batch_size)
        attribution[i], outputs[i] = _calc_gradients(model=model, data=_dataset,
                                      inference_fn=inference_fn, device=device, return_outputs=True)
    model.train(_prev_train_state)

    # scale gradients according to distance between point and previous point
    if hasattr(scaling, 'shape') and scaling.shape == paths.shape[:-1]:
        for _ in range(paths.shape[0]):
            attribution[0] = attribution[0] * scaling[0].unsqueeze(-1)
    else:
        if np.isscalar(scaling):
            scaling = torch.full((n_samples,), scaling)
        if scaling.shape[:2] != (input.shape[0], n_samples):
            for _ in range(len(paths.shape[2:])):
                scaling = scaling.unsqueeze(-1)
            for i in range(attribution.shape[0]):
                attribution[i] *= scaling

    # sum up over path
    attribution = torch.sum(attribution, dim=1)

    if simplified:
        attribution = torch.sum(attribution, dim=-1)

    if _use_tuple:
        attribution = (attribution,)  # OMG DUH GRHNG

    _numerical_delta = torch.nan
    if outputmode == 'full' or return_convergence_delta:

        at_baseline = torch.gather(outputs[:, 0], -1, targets.unsqueeze(1)).squeeze()
        at_sample = torch.gather(outputs[:, -1], -1, targets.unsqueeze(1)).squeeze()
        diff = at_sample - at_baseline
        # for tabular data -> obtain attribution mass of each sample
        _global_attrs = attribution * (data - baselines)
        _sums = torch.sum(_global_attrs, -1)
        # for non-tabular data we need to keep summing
        while len(_sums.shape) > 1:
            _sums = torch.sum(_sums, -1)
        _numerical_delta = torch.abs(diff - _sums)
        # perc = diff / _sums
        # _delta_percent = (1.0 - perc.abs()).abs()
        # print(torch.sum(torch.le(_delta_percent, 0.05)))
        # assert torch.all(torch.le( _delta_percent, 0.05))  # check if percentage of difference is less than 5% for all samples

    if outputmode == 'full':
        return attribution, paths, scaling, _numerical_delta

    if return_convergence_delta:
        return attribution, _numerical_delta

    return attribution


def left_integrated_gradients(model, data, targets=None, inference_fn=None, baselines=None, steps=150, simplified=False,
                         calc_paths=None, calc_baselines=None, fit_baseline_data=False, device=None, outputmode=None,
                              alpha=.9, needs_softmax=True):
    """
    From
    **Miglani, V., Kokhlikyan, N., Alsallakh, B., Martin, M., & Reblitz-Richardson, O. (2020).
    Investigating Saturation Effects in Integrated Gradients. Whi. http://arxiv.org/abs/2010.12697**

    :param model:
    :param data:
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: if none given, zero baseline is used,
        "integral approximated with left riemannian integration plus inclusion of final point",
        ie we interpolate linearly , x0 + t(x - x0) t = [0., ..., 1.], and include start and end points
    :param steps: number of steps between baseline and input -> actual steps = steps+2; only used when calc_paths is not a callable
    :param simplified: return attribution of shape (1, seqlen), accumulate attribution for each vector
            if false, returns all collected gradients of size [n_samples, n_steps, seqlen, embedding_size]
    :param calc_paths: needs to return paths between all samples as well as distance between consecutive samples
    :param calc_baselines: must return one baseline per sample
    :return:
    """


    if device is None:
        device = next(model.parameters()).device  #  this looks ugly but is apparently the way to go in vanilla pytorch

    if inference_fn is None:
        inference_fn = model

    _use_tuple = isinstance(data, tuple)
    ## put everything on cpu for now
    if _use_tuple:
        data = data[0].to('cpu')
    else:
        data = data.to('cpu')

    input = data

    # support multiple baselines per sample?
    if callable(calc_baselines):
        baselines = calc_baselines(data) # (samples, model)?
    if baselines is None:
        baselines = torch.zeros_like(input)
    elif fit_baseline_data:
        baselines = baselines.repeat(input.shape[:-1] + (1,))


    # paths.size = [n_samples, n_steps, seqlen, embedding]
    paths = None
    if callable(calc_paths):
        paths = calc_paths(baselines, input, steps)
    else:
        # fallback: straight line between baseline and input,
        p = np.linspace(baselines, input, num=steps)#, retstep=True)
        paths = torch.tensor(p).swapaxes(0, 1).requires_grad_()

    assert paths is not None

    if targets is None:
        predictions = _get_targets(inference_fn, data, model, device)
    else:
        predictions = targets

    # places gradients in attribution
    attribution = torch.zeros_like(paths)
    outputs = []
    for i in range(paths.shape[0]):
        path = paths[i]
        prediction = predictions[i].expand(paths.shape[1])
        _dataset = DataLoader(TensorDataset(path, prediction), shuffle=False, batch_size=paths.shape[0])
        attribution[i], _outputs = _calc_gradients(model=model, data=_dataset,
                                      inference_fn=inference_fn, device=device, return_outputs=True)
        outputs.append(_outputs)

# --- CALCULATE WHAT IS "LEFT"
    # outputs.shape = (N, steps, n_classes)
    outputs = torch.stack(outputs)
    if needs_softmax:
        outputs = torch.softmax(outputs, dim=-1)
    # throw all outputs except for the target class away
    outputs = torch.stack([outputs[i, :, t] for i, t in enumerate(targets)]) # shape (samples, steps, 1)
    # get confidence at sample itself
    conf = outputs[:, -1] # shape (samples, 1)
    threshold = alpha * conf
    threshold = torch.reshape(torch.repeat_interleave(threshold, steps), shape=outputs.shape) # shape (samples, steps, 1)
    mask = outputs <= threshold # shape (samples, steps)
    '''
    _each_ step that is below threshold is further considered, regardless of whether some prior steps might
    have been below threshold already;
    also: during testing I observed cases where outputs[i, -1] was not only not the maxmium, it acutally was 
    the _minimum_ -> meaning the mask zero'd out the whole attribution sequence
    '''
    # shape (samples, steps, *data.shape)
    mask = torch.reshape(mask, shape=(*mask.shape, *(1,)*len(attribution.shape[2:])))
    # all attributions at steps where outputs were below threshold will be kept, all other set to 0
    attribution = mask * attribution

    attribution = torch.sum(attribution, dim=1)

    if simplified:
        attribution = torch.sum(attribution, dim=-1)

    if _use_tuple:
        attribution = (attribution,)

    if outputmode == 'full':
        return attribution, paths, threshold, mask

    return attribution



def compute_output_curves(model, sample, target, baselines, inference_fn=None, steps=150,
                  saturation_threshold=.9, device=None, return_grads=False,
                  ):
    ''' computes outputs on paths from **INPUT TO BASELINE** '''

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model

    p = np.linspace(sample, baselines, num=steps)#, retstep=True)
    paths = torch.tensor(p).swapaxes(0, 1).requires_grad_()

    outputs = []
    if return_grads:
        gradients = []
        targets = target.expans(paths.shape[1])

    for i, path in enumerate(paths):
        if return_grads:
            _data = DataLoader(TensorDataset(path, targets), shuffle=False, batch_size=paths.shape[1])
            grads, outs = _calc_gradients(model, _data, inference_fn, device, return_outputs=True)
            gradients.append(grads)
            outputs.append(outs[:, target])
        else:
            outs = _get_outputs(inference_fn, path, model, device)[:, target]
            outputs.append(outs)


    outputs = torch.stack(outputs)
    if return_grads:
        gradients = torch.stack(gradients)
        return outputs, gradients

    return outputs


class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)


def _check_baselines(baselines):
    # make it iterable if it is a scalar

    if not hasattr(baselines, '__iter__'):
        return [baselines]
    return baselines


def kernelshap(model, data, targets=None, inference_fn=None, baselines=None, device=None,
             n_samples=None, masks=None):
    """
    **tested for tabular data only so far**
    uses lime with appropriate loss function (MSE), weighting kernel (Shapley Kernel) and regularization (ie none)
    :param model: model
    :param data: DataLoader
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: defaultis None; scalar, single baseline or one baseline per input
    :param n_samples: default 25; number of samples to train regression on
    :return: Attribution scores computed by KernelShap

    """

    if device is not None:
        model = model.to(device)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model

    if n_samples is None:
        n_samples = 2 * data[0].shape[0]  # TODO: verify choice; on NLP with sentence len=93 for n=25 argsort was unstable, n=200 was 'better'

    # attributions = []
    # baselines = _check_baselines(baselines)
    # if masks is None:
    #     masks = [None]
    # for x, b, y, m in zip(data, cycle(baselines), targets, cycle(masks)):
    explainer = KernelShap(inference_fn)
    data = data.to(device)
    if targets is not None:
        targets = targets.to(device)
    if masks is not None:
        masks = masks.to(device)
    _prev_train_state = model.training
    model.train(True)
    attributions = explainer.attribute(data, baselines, targets, n_samples=n_samples, feature_mask=masks)
    model.train(_prev_train_state)

    return attributions

def lime(model, data, targets=None, inference_fn=None, baselines=None, device=None,
             n_samples=25, distance_mode="cosine"):
    """
    **tested for tabular data only so far**

    :param model: model
    :param data: input data, expects first dim to be batchsize
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: defaultis None; scalar, single baseline or one baseline per input
    :param n_samples: default 25; number of samples to train regression on
    :param distance_mode: default "cosine"; distance mode to be used in loss for surrogate model, alternatively "euclidean"
    :return: Attribution scores computed by Lime
    """

    if device is not None:
        model = model.to(device)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model


    from captum.attr._core.lime import get_exp_kernel_similarity_function
    similarity_fn = get_exp_kernel_similarity_function(distance_mode=distance_mode)

    explainer = Lime(inference_fn, similarity_func=similarity_fn)

    data = data.to(device)
    if targets is not None:
        targets = targets.to(device)
    _prev_train_state = model.training
    model.train(True)
    attributions = explainer.attribute(data, baselines, targets, n_samples=n_samples)  # n_samples=25 provided the same ranking for top 10 features as n=250
    model.train(_prev_train_state)
    return attributions

