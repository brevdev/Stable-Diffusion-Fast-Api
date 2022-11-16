import requests
import base64

headers = {
    # requests won't add a boundary if this header is set when you pass files=
    # 'Content-Type': 'multipart/form-data',
}

# files = {
#     '{     "prompt":"snoop dogg on the moon",      "seed": 4422,   "num_outputs": 1,       "width": 256,   "height": 256,  "num_inference_steps": 50,\n"guidance_scale": 7}': (None, ''),
#     'prompt': (None, 'snoop dogg on mars'),
# }
files = {
    'prompt': (None, 'snoop dogg wearing sunglasses'),
}
response = requests.get('http://127.0.0.1:1337/txt2img?prompt=snoop+dog+with+sunglasses', headers=headers) #, files=files
# response = requests.get('http://127.0.0.1:1337/txt2img', headers=headers, files=files)
print("response is: ", response)
# print parameters of response:
print("response.text is: ", response.text)
image = response.json()['images'][0]
print("images is: ", image)
base64text = image['base64']
imgdata = base64.b64decode(base64text)

# save the image to a file:
with open("image.png", "wb") as fh:
    fh.write(imgdata)
    
print("image['base64'] is: ", image['base64'])