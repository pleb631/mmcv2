import torch
from torch import Tensor


def scatter(input: list | Tensor, devices: list, streams: list | None = None) -> list | Tensor:
    """Scatters tensor across multiple GPUs."""
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [scatter(input[i], [devices[i // chunk_size]], [streams[i // chunk_size]]) for i in range(len(input))]
        return outputs
    elif isinstance(input, Tensor):
        output = input.contiguous()
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            if output.device.type == "cpu" and not output.is_pinned():
                output = output.pin_memory()
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)

        return output


def synchronize_stream(output: list | Tensor, devices: list, streams: list) -> None:
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]], [streams[i]])
    elif isinstance(output, Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)


def get_input_device(input: list | Tensor) -> int:
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, Tensor):
        return input.get_device() if input.is_cuda else -1


class Scatter:
    @staticmethod
    def forward(target_gpus: list[int], input: list | Tensor) -> tuple:
        input_device = get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            streams = [torch.cuda.Stream(device=device) for device in target_gpus]

        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs,)
