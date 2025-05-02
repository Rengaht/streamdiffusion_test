import torch
import SpoutGL
from itertools import islice, cycle, repeat
import array
from random import randint
import time
from OpenGL import GL

from multiprocessing import Queue
import numpy as np

TARGET_FPS = 30
SEND_WIDTH = 512
SEND_HEIGHT = 512


def spout_buffer_to_tensor(buffer, width, height):
    np_buffer = np.asarray(buffer, dtype=np.uint8)
    image_bgra = np_buffer.reshape((height, width, 4))

    image_rgb = image_bgra[..., [2, 1, 0]]
    image_float = image_rgb.astype(np.float32) / 255.0
    image_normalized = (image_float * 2.0) - 1.0
    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)

    return tensor.unsqueeze(0)


def get_spout_image(queue, wwidth: int, wheight: int) -> None:
    with SpoutGL.SpoutReceiver() as receiver:
        receiver.setReceiverName("SpoutSender")

        buffer = None

        while True:
            result = receiver.receiveImage(buffer, GL.GL_RGBA, False, 0)

            if receiver.isUpdated():
                width = receiver.getSenderWidth()
                height = receiver.getSenderHeight()
                buffer = array.array('B', [0] * (width * height * 4))  # Correctly reallocate buffer with updated size

            if buffer and result and not SpoutGL.helpers.isBufferEmpty(buffer):
                pixels=spout_buffer_to_tensor(buffer, width, height)
                queue.put(pixels, block=False)

            # Wait until the next frame is ready
            # Wait time is in milliseconds; note that 0 will return immediately
            receiver.waitFrameSync("SpoutSender", 10000)
        
    


def randcolor():
    return randint(0, 255)


def tensor_to_spout_image(tensor):
    image = tensor.squeeze(0)
    image = image.permute(1, 2, 0)
    image_np = image.cpu().numpy()

    if image_np.min() < 0:
        image_np = (image_np + 1) / 2  # Scale from [-1, 1] to [0, 1]
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    h, w, _ = image_np.shape
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    image_rgba = np.concatenate((image_np, alpha), axis=-1)

    image_bgra = image_rgba[..., [2, 1, 0, 3]]

    return np.ascontiguousarray(image_bgra)  # Ensure the array is contiguous in memory

def send_spout_image(queue: Queue, width: int, height: int)->None:
    
    with SpoutGL.SpoutSender() as sender:
        sender.setSenderName("StreamDiffusion")
    
        while True:

            # Check if there are images in the queue
            if not queue.empty():
                image = queue.get(block=False)
                pixels = tensor_to_spout_image(image)

                result = sender.sendImage(pixels, width, height, GL.GL_RGBA, False, 0)
                # print("Send result", result)
                
                # Indicate that a frame is ready to read
                sender.setFrameSync("StreamDiffusion")
                
                # Wait for next send attempt
                time.sleep(1./TARGET_FPS)