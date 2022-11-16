curl --request POST \              ~/Desktop
  --url http://127.0.0.1:1337/txt2img \
  --header 'Content-Type: multipart/form-data' \
  --form '{     "prompt":"snoop dogg on the moon",      "seed": 4422,   "num_outputs": 1,       "width": 256,   "height": 256,  "num_inference_steps": 50,
"guidance_scale": 7}=' \
  --form 'prompt=snoop dogg on mars' 