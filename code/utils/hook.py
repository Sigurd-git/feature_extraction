from torchaudio.models.wav2vec2.components import ConvLayerBlock


def hook_fn(module, input, output, activations):
    activations.append(output)


def register_activation_hooks(model, layer_type=ConvLayerBlock):
    """Register forward hooks for all convolutional layers in the model to capture activations.

    Parameters:

    model: PyTorch model.

    layer_type: The type of layer to register hooks for. Default: ConvLayerBlock.


    Returns:

    hooks: A list of hook handles.
    activations: A empty list to store activations.
    """
    hooks = []
    activations = []

    def hook_fn_wrapper(module, input, output):
        return hook_fn(module, input, output, activations)

    for layer in model.modules():
        if isinstance(layer, layer_type):
            hook = layer.register_forward_hook(hook_fn_wrapper)
            hooks.append(hook)
    return hooks, activations


def remove_hooks(hooks):
    """
    Remove all hooks registered through register_activation_hooks.

    Parameters:

    hooks: A list of hook handles returned by register_activation_hooks.
    """
    for hook in hooks:
        hook.remove()
