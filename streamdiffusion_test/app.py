import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from StreamDiffusion.utils.viewer import receive_images

from StreamDiffusion.utils.wrapper import StreamDiffusionWrapper
from threading import Thread

import os
import sys
import time
from multiprocessing import Process, Queue, get_context

import fire
import torchvision.transforms as transforms

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    prompt: str,
    model_id_or_path: str,
)-> None:
    # stream = StreamDiffusionWrapper(       
    #         model_id_or_path=model_id_or_path,
    #         lora_dict=None,
    #         t_index_list=[0, 16, 32, 45],
    #         frame_buffer_size=1,
    #         width=512,
    #         height=512,
    #         warmup=10,
    #         acceleration="xformers",
    #         mode="txt2img",
    #         use_denoising_batch=False,
    #         cfg_type="none",
    #         seed=2,
    #     )
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0],
        frame_buffer_size=1,
        warmup=10,
        acceleration="tensorrt",
        use_lcm_lora=False,
        mode="txt2img",
        cfg_type="none",
        use_denoising_batch=True,
    )
     
    # prompt = "A glowing, vintage phone booth standing in surreal landscapes across different scene"
    # Prepare the stream
    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    # Prepare image
    # init_image = load_image("example.png").resize((512, 512))

    # Warmup >= len(t_index_list) x frame_buffer_size
    # for _ in range(stream.batch_size - 1):
    #     stream()

    previous_output = None
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    while True:
        try:
            start_time = time.time()
            # x_output = stream(image=previous_output)
            x_output=stream.stream.txt2img_sd_turbo(1).cpu()
            # x_output_tensor = transform(x_output)
            queue.put(x_output.cpu(), block=False)

            # Calculate FPS
            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
            
            previous_output = x_output

        except KeyboardInterrupt:
            print(f"fps: {fps}")
            return

def main()-> None:

    try:
        ctx = get_context('spawn')
        queue = Queue()
        fps_queue = Queue()

        prompt = "A glowing, vintage phone booth standing in surreal landscapes across different scene"
        # model_id_or_path = "KBlueLeaf/kohaku-v2.1"
        model_id_or_path = "stabilityai/sd-turbo"


        process1= ctx.Process(
            target=image_generation_process,
            args=(queue, fps_queue, prompt, model_id_or_path),
        )
        process1.start()

        process2=ctx.Process(target=receive_images, args=(queue, fps_queue))
        process2.start()

        process1.join()
        process2.join()
    except KeyboardInterrupt:
        print("Process interrupted")
        
        process1.terminate()
        process2.terminate()
        return



if __name__ == "__main__":
    fire.Fire(main)
