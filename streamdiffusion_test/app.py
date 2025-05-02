import os
import sys
import time

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

sys.path.insert(0, os.path.abspath('../StreamDiffusion'))

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from utils.viewer import receive_images

from utils.wrapper import StreamDiffusionWrapper
from threading import Thread


from multiprocessing import Process, Queue, get_context

from perlin import perlin_2d, rand_perlin_2d, rand_perlin_2d_octaves, perlin_2d_octaves
from scene_prompt import surreal_prompt_parts
from scene_prompt import surreal_prompts
from scene_prompt import regret_prompts

from spout_util import send_spout_image, get_spout_image


import fire

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    noise_queue: Queue,
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
    idx=0
    last_time = time.time()

    while True:
        # try:
        start_time = time.time()
        # x_output = stream(image=previous_output)
        # x_output=stream.stream.txt2img_sd_turbo(1).cpu()

      

        noise= noise_queue.get(block=True)

        if int(time.time()) % 5 == 0 and last_time != int(time.time()) and idx<len(regret_prompts)-1:
            # idx=int(time.time()) % len(surreal_prompt_parts) 
            idx=min((idx+1), len(regret_prompts)-1)
            tmp_prompt= ", ".join(regret_prompts[:idx+1])

            # idx=int(time.time()) % len(surreal_prompts)
            # idx=(idx+1)%len(surreal_prompts)
            # tmp_prompt=surreal_prompts[idx]

            # last_frame=torch.clamp(previous_output, 0, 1)
            x_output=stream.img2img(image=noise, prompt="a surreal landscape, "+tmp_prompt)

            print(f"update prompt: {tmp_prompt}")
            last_time = int(time.time())
        else:
            x_output=stream.img2img(image=noise)
        
        # if isinstance(x_output, torch.Tensor) and x_output.dim() == 3:
        #     x_output = x_output.permute(1, 2, 0)  # Convert from C x H x W to H x W x C
        # x_tensor_output = transforms.ToTensor()(x_output)

        preprocessed_image =stream.preprocess_image(x_output)
       
        queue.put(preprocessed_image, block=False)

        # queue.put(preprocessed_image, block=False)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        fps_queue.put(fps)
        
        # x_output = (x_output + 1) / 2  # Scale from [-1, 1] to [0, 1]
        # x_output = torch.clamp(x_output, 0, 1)
        previous_output = x_output

        # except KeyboardInterrupt:
        #     print(f"fps: {fps}")
        #     return

def noise_generation(queue: Queue, width: int, height: int, batch_size: int = 1) -> None:
    while True:
        # try:
        # Generate random noise
        # noise_images = torch.randn(batch_size, 3, height, width).uniform_(0,1)
        
        # perlin_images = rand_perlin_2d_octaves((height, width), (8, 8))
        perlin_images=perlin_2d_octaves((height, width), (2, 2), time.time()*5,4)
        perlin_images = torch.tensor(perlin_images).unsqueeze(0).repeat(batch_size, 3, 1, 1)
        # Normalize perlin_images to the range [0, 1]
        perlin_images=(perlin_images+1)/2
        perlin_images = torch.clamp(perlin_images, 0, 1)


        queue.put(perlin_images, block=False)
        # except KeyboardInterrupt:
        #     print("Noise generation interrupted")
        #     return


def main()-> None:

    try:
        ctx = get_context('spawn')
        queue = Queue()
        fps_queue = Queue()
        noise_queue = Queue()

        # prompt = "A surreal landscapes"
        prompt=regret_prompts[0]
        # model_id_or_path = "KBlueLeaf/kohaku-v2.1"
        model_id_or_path = "stabilityai/sd-turbo"


        # Start the noise generation process
        process_noise = ctx.Process(
            # target=noise_generation,
            target=get_spout_image,
            args=(noise_queue, 512, 512),
        )
        process_noise.start()


        process_gen= ctx.Process(
            target=image_generation_process,
            args=(queue, fps_queue, noise_queue, prompt, model_id_or_path),
        )
        process_gen.start()



        # process_show=ctx.Process(target=receive_images, args=(queue, fps_queue))
        # process_show.start()


        process_spout=ctx.Process(target=send_spout_image, args=(queue, 512, 512))
        process_spout.start()


        process_noise.join()
        # process_gen.join()        
        # process_show.join()
        process_spout.join()


    except KeyboardInterrupt:
        print("Process interrupted")
        
        # process_gen.terminate()
        # process_show.terminate()
        process_noise.terminate()
        process_spout.terminate()

        return



if __name__ == "__main__":
    fire.Fire(main)
