"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""
import base64
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionXLPipeline

class Model:
    def __init__(self, **kwargs):
        #Basten Simple file 
        # Uncomment the following to get access
        # to various parts of the Truss config.
        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None
        self.diffusion_model_path = "SG161222/RealVisXL_V4.0"
        self.cache_dir = None
        self.precision = "fp16"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        # Load model here and assign to self._model.
        print("Loading models...")
       
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.diffusion_model_path,
                cache_dir = self.cache_dir,
                variant =self.precision,
                torch_dtype = torch.float16,
            ).to(self.device)
        print("Models loaded successfully.")

    def preprocess(self, request):
        generate_args = {
            "prompts": request.get('prompts', [""]),
            "seed": request.get('seed', 1234),
            "num_inference_steps":request.get('num_inference_steps', 30),
            "num_per_prompt":request.get('num_per_prompt', 1),
            "negative_prompt":request.get('negative_prompt', [""]),
        }
        print("Generate arguments:", generate_args)
        request["generate_args"] = generate_args
        return request

    def predict(self, request):
        # Run model inference here
        model_input = request.pop("generate_args")
        prompts = model_input['prompts']
        num_per_prompt = model_input['num_per_prompt']
        negative_prompt = model_input['negative_prompt']
        num_inference_steps = model_input['num_inference_steps']
        seed = model_input['seed']
        try:
            images  = self.sdxl_pipe(
                prompt=prompts,     
                negative_prompt=negative_prompt, 
                generator=torch.manual_seed(seed),
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_per_prompt,
                ).images
            print("Pipeline results obtained successfully.")
        except Exception as e:
            print("Error during model inference:", e)
            return {'error': str(e)}

        # Encode the images
        encoded_output_images = []
        for label, image in enumerate(images):
            print(f"Processing result image with label: {label}")
            try:
                buffered = BytesIO()
                image.save(buffered, format='PNG')
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                encoded_output_images.append({
                    'label': label,
                    'image': encoded_image
                })
                print(f"Image with label '{label}' encoded successfully.")
            except Exception as e:
                print(f"Error encoding image with label '{label}':", e)

        return {'output_images': encoded_output_images}
