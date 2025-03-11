# Zero-Cost LLM Hosting: Unleash vLLM Locally

Here is the repository that contains the demo scripts of this [Rabiloo blog post](https://rabiloo.com/blog/deploying-local-llm-hosting-for-free-with-vllm).

We are going to follow the instructions below to set up the environment and serve LLMs locally.

## Prerequisites
- 🐍Python 3.8+ 
- 💡 A basic understanding of Python (if you can write a print('Hello, world!'), you’re in)
- 🚀 (Optional) A modern GPU (ideally with 16GB+ VRAM, but lower-end ones might work too) 
- ⚡ (Optional) If you want to take advantage of GPU, you should have NVIDIA CUDA installed (check your version: ```nvidia-smi```)

## Serve LLMs
 First, navigate to the vllm folder, then run the following command:

```bash
cd vllm/scripts
```

### Choose your LLM

Pick the LLM model you want to serve. Find the Right Model Name on Hugging Face
To ensure compatibility, you need the correct model name from Hugging Face Model Hub:
- Visit [Hugging Face](https://huggingface.co/).
- Search for a model (e.g., Llama 3, Mistral, Qwen).
- Click on the model you want.
- Copy the full repository name from the page URL, typically in org_name/model_name format. Example: Qwen/Qwen2-1.5BB-Instruct or microsoft/Magma-8B.

**Note:** You can use your fine-tuned LLM (with LoRA, Deepspeed, or something else). And we will talk about how to fine-tune LLM in the LLM fine-tuning series. 

### Docker

If you have Docker installed and your own GPU, you can start serving LLMs with this command:

```bash
bash get_started_gpu_docker.sh
```

You can read the ```get_started_gpu_docker.sh``` file or the blog post for more information.

Check the logs of the Docker container:

```bash
docker logs -f <docker-container-id>
````

In case you don’t have a GPU or just want to test an LLM on your CPU, run this:

```bash
bash get_started_cpu.sh
```

If you see logs like this:

```
INFO: 	Started server process [1]
INFO: 	Waiting for application startup.
INFO: 	Application startup complete.
```

It means that the server is running successfully.

### Python

Run this command to start serving LLMs with Python:

```bash
bash get_started_gpu_python.sh
```

## Integrate the Model into Applications

### FastAPI
Run the command:

```bash
bash scripts/fastapi.sh
```

The app will be available at http://127.0.0.1:8000, and we can try it with Swagger UI at http://127.0.0.1:8000/docs.

Stop the app: Press ```Ctrl+C```

### Gradio
Run the command:

```bash
bash scripts/gradio.sh
```

You can try the gradio app at:
- local URL http://0.0.0.0:7860
- public URL: https://xxxxxxxxxxxxxxxxxx.gradio.live

(URLs will be shown in the logging)

Stop the app: Press ```Ctrl+C```

## Tip: Run vLLM with GPU on Google Colab
Check out notebook vllm_colab.ipynb and [blog post](https://rabiloo.com/blog/deploying-local-llm-hosting-for-free-with-vllm) for modification.

## References
- [vLLM](https://github.com/vllm-project/vllm)

## Contact
If you have any questions, or you want to collaborate with us, please feel free to contact us via https://rabiloo.com/contact-us.
