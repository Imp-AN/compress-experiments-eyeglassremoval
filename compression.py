import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def global_unstructured_prune(model, amount):
    """Globally prune the smallest-magnitude individual convolution weights."""
    model = copy.deepcopy(model)

    parameters = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            parameters.append((module, "weight"))

    prune.global_unstructured(
        parameters,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def structured_prune(model, amount):
    """Per Conv2d layer, prune output filters with the smallest L2 norm."""
    model = copy.deepcopy(model)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0,
            )
    return model


def apply_pruning(model, config):
    if config["kind"] == "unstructured":
        return global_unstructured_prune(model, config["amount"])
    if config["kind"] == "structured":
        return structured_prune(model, config["amount"])
    raise ValueError(f"Unknown pruning kind: {config['kind']}")


def finetune(model, train_loader, device, epochs=1, lr=1e-5):
    """Let surviving weights adapt while pruning masks keep pruned weights at zero."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for _ in range(epochs):
        model.train()
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)

            pred = model(x)
            loss = F.l1_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def static_int8_quantize(model, calibration_loader):
    """PT2E static INT8 quantization for x86 CPU using representative calibration data."""
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
    import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq

    model = copy.deepcopy(model).cpu().eval()

    example_x, _ = next(iter(calibration_loader))
    example_inputs = (example_x[:1].cpu(),)

    exported = export(model, example_inputs).module()

    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    prepared = prepare_pt2e(exported, quantizer)

    with torch.no_grad():
        for x, _ in calibration_loader:
            prepared(x.cpu())

    return convert_pt2e(prepared)
