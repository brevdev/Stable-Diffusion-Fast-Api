This repo will run a fast api for text to image Stable Diffusion. Most of the work here was done by [Christian Cantrell](https://github.com/cantrell). Please check him out!

# To run on your own machine:

```
conda env create -f environment.yaml
conda activate sd-api-server
uvicorn server:app --reload --host 0.0.0.0
```

If you don't have conda installed, you can use pip with:

```
!python3 -m venv sd-api-server-env
!source sd-api-server-env/bin/activate
!pip install -r requirements.txt -q -U/
```

This will create a virtual environment with all the dependencies you need and then run the server. If you don't have the Stable Diffusion model downloaded, this will download it before running the server.

# To run on Brev:
[![](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/pill-border-lg.png)](https://console.brev.dev/environment/new?repo=https://github.com/brevdev/Stable-Diffusion-Fast-Api.git&instance=g5.xlarge&diskStorage=60)


Create your environment by following this [link](https://console.brev.dev/environment/new?repo=https://github.com/brevdev/Stable-Diffusion-Fast-Api.git&instance=g5.xlarge&diskStorage=60) pre-populated with config you need.

[Install the Brev CLI](https://brev.dev/docs/how-to/install-cli) and open your Stable Diffusion Fast Api environmnent:
```
brev open stable-diffusion-fast-api --wait
```
Then activate your environment and run the server:
```
conda activate sd-api-server
uvicorn server:app --reload --host 0.0.0.0
```

Then to make requests:

1. Forward port 8000 in VScode
2. Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) or make calls through [http://127.0.0.1:8000/txt2img?prompt=Snoop+Dogg](http://127.0.0.1:8000/txt2img?prompt=Snoop+Dogg)

# Public ports:
With Brev exposing ports publically is super easy! 
Just go to [console.brev.dev](https://console.brev.dev/) and under Environment Settings you'll be able to expose ports publically. Make sure you expose port 8000!

