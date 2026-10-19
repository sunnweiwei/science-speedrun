"""Use norm32 on every CUDA GPU; eager remains an explicit override."""
BACKENDS = ('eager', 'norm32', 'norm32-graph')


def select_backend(model, requested):
    """Install norm32 atomically; dependency, parity and execution errors propagate."""
    if requested not in BACKENDS:
        raise ValueError('Unknown inference backend')
    if requested == 'norm32':
        from ffn_inference import install_norm32
        install_norm32(model)
    return requested


def receipt_backend_matches(receipt, requested):
    actual = receipt.get('inference_backend', 'eager')
    recorded_request = receipt.get('requested_inference_backend', actual)
    return requested in BACKENDS and recorded_request == requested and actual == requested
