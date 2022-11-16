import io
import os
import re
import time
import inspect
import json
from fastapi import FastAPI
import flask
import sys
import base64
from PIL import Image
from io import BytesIO
from starlette.responses import StreamingResponse
from typing import Union

import torch
import diffusers

app = FastAPI()

##################################################
# Utils

def retrieve_param(key, data, cast, default):
    if key in data:
        value = flask.request.form[ key ]
        value = cast( value )
        return value
    return default

def pil_to_b64(input):
    buffer = BytesIO()
    input.save( buffer, 'PNG' )
    output = base64.b64encode( buffer.getvalue() ).decode( 'utf-8' ).replace( '\n', '' )
    buffer.close()
    return output

def b64_to_pil(input):
    output = Image.open( BytesIO( base64.b64decode( input ) ) )
    return output

def get_compute_platform(context):
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available() and context == 'engine':
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

##################################################
# Engines

class Engine(object):
    def __init__(self):
        pass

    def process(self, kwargs):
        return []

class EngineStableDiffusion(Engine):
    def __init__(self, pipe, sibling=None, custom_model_path=None, requires_safety_checker=True):
        super().__init__()
        if sibling == None:
            self.engine = pipe.from_pretrained( 'runwayml/stable-diffusion-v1-5', use_auth_token=hf_token.strip() )
        elif custom_model_path:
            if requires_safety_checker:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                safety_checker=sibling.engine.safety_checker,
                                                                                feature_extractor=sibling.engine.feature_extractor)
            else:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                feature_extractor=sibling.engine.feature_extractor)
        else:
            self.engine = pipe(
                vae=sibling.engine.vae,
                text_encoder=sibling.engine.text_encoder,
                tokenizer=sibling.engine.tokenizer,
                unet=sibling.engine.unet,
                scheduler=sibling.engine.scheduler,
                safety_checker=sibling.engine.safety_checker,
                feature_extractor=sibling.engine.feature_extractor
            )
        self.engine.to( get_compute_platform('engine') )

    def process(self, kwargs):
        output = self.engine( **kwargs )
        return {'image': output.images[0], 'nsfw':output.nsfw_content_detected[0]}

class EngineManager(object):
    def __init__(self):
        self.engines = {}

    def has_engine(self, name):
        return ( name in self.engines )

    def add_engine(self, name, engine):
        if self.has_engine( name ):
            return False
        self.engines[ name ] = engine
        return True

    def get_engine(self, name):
        if not self.has_engine( name ):
            return None
        engine = self.engines[ name ]
        return engine

##################################################
# App

if not os.path.isfile('config.json'):
    print('Please enter your HuggingFace token available here https://huggingface.co/settings/tokens:')
    huggingface_token = input()
    # write token to config.json:
    with open('config.json', 'w') as f:
        f.write('{"hf_token": "' + huggingface_token + '"}')
# Load and parse the config file:
try:
    config_file = open ('config.json', 'r')
except:
    sys.exit('config.json not found.')

config = json.loads(config_file.read())

hf_token = config['hf_token']

if (hf_token == None):
    sys.exit('No Hugging Face token found in config.json.')

custom_models = config['custom_models'] if 'custom_models' in config else []

# Initialize app:
# app = flask.Flask( __name__ )

# Initialize engine manager:
manager = EngineManager()

# Add supported engines to manager:
manager.add_engine( 'txt2img', EngineStableDiffusion( diffusers.StableDiffusionPipeline,        sibling=None ) )
manager.add_engine( 'img2img', EngineStableDiffusion( diffusers.StableDiffusionImg2ImgPipeline, sibling=manager.get_engine( 'txt2img' ) ) )
manager.add_engine( 'masking', EngineStableDiffusion( diffusers.StableDiffusionInpaintPipeline, sibling=manager.get_engine( 'txt2img' ) ) )
for custom_model in custom_models:
    manager.add_engine( custom_model['url_path'],
                        EngineStableDiffusion( diffusers.StableDiffusionPipeline, sibling=manager.get_engine( 'txt2img' ),
                        custom_model_path=custom_model['model_path'],
                        requires_safety_checker=custom_model['requires_safety_checker'] ) )

# Define routes:
# @app.route('/ping', methods=['GET'])
# def stable_ping():
#     return flask.jsonify( {'status':'success'} )

# @app.route('/custom_models', methods=['GET'])
# def stable_custom_models():
#     if custom_models == None:
#         return flask.jsonify( [] )
#     else:
#         return custom_models

# @app.route('/txt2img', methods=['GET'])
@app.get("/txt2img")
def stable_txt2img(prompt: Union[str, None]):
    print("input prompt is: ", prompt)
    return _generate('txt2img', prompt)

# @app.route('/txt2img', methods=['POST'])
# def stable_txt2img():
#     return _generate('txt2img')



# @app.route('/img2img', methods=['POST'])
# def stable_img2img():
#     return _generate('img2img')

# @app.route('/masking', methods=['POST'])
# def stable_masking():
#     return _generate('masking')

# @app.route('/custom/<path:model>', methods=['POST'])
# def stable_custom(model):
#     return _generate('txt2img', model)

def _generate(task, prompt="Snoop Dogg", engine=None):
    # Retrieve engine:
    if engine == None:
        engine = task

    engine = manager.get_engine( engine )

    # Prepare output container:
    output_data = {}
    # Handle request:
    try:
        seed = 0#retrieve_param( 'seed', flask.request.form, int, 0 )
        count = 1#retrieve_param( 'num_outputs', flask.request.form, int,   1 )
        total_results = []
        for i in range( count ):
            if (seed == 0):
                generator = torch.Generator( device=get_compute_platform('generator') )
            else:
                generator = torch.Generator( device=get_compute_platform('generator') ).manual_seed( seed )
            new_seed = generator.seed()
            args_dict = {
                'prompt' : [ prompt ],
                'num_inference_steps' : 100, #retrieve_param( 'num_inference_steps', flask.request.form, int,   100 ),
                'guidance_scale' : 7.5, #retrieve_param( 'guidance_scale', flask.request.form, float, 7.5 ),
                'eta' : 0.0, #retrieve_param( 'eta', flask.request.form, float, 0.0 ),
                'generator' : generator
            }
            if (task == 'txt2img'):
                args_dict[ 'width' ] = 512 #retrieve_param( 'width', flask.request.form, int,   512 )
                args_dict[ 'height' ] = 512 #retrieve_param( 'height', flask.request.form, int,   512 )
            # if (task == 'img2img' or task == 'masking'):
            #     init_img_b64 = flask.request.form[ 'init_image' ]
            #     init_img_b64 = re.sub( '^data:image/png;base64,', '', init_img_b64 )
            #     init_img_pil = b64_to_pil( init_img_b64 )
            #     args_dict[ 'init_image' ] = init_img_pil
            #     args_dict[ 'strength' ] = 0.7 #retrieve_param( 'strength', flask.request.form, float, 0.7 )
            # if (task == 'masking'):
            #     mask_img_b64 = flask.request.form[ 'mask_image' ]
            #     mask_img_b64 = re.sub( '^data:image/png;base64,', '', mask_img_b64 )
            #     mask_img_pil = b64_to_pil( mask_img_b64 )
            #     args_dict[ 'mask_image' ] = mask_img_pil
            # Perform inference:
            print("ARGS DICT IS: ", args_dict)
            pipeline_output = engine.process( args_dict )
            pipeline_output[ 'seed' ] = new_seed
            total_results.append( pipeline_output )
        # Prepare response
        output_data[ 'status' ] = 'success'
        images = []
        print("totla results length: ", len(total_results))
        for result in total_results:
            images.append({
                'base64' : pil_to_b64( result['image'].convert( 'RGB' ) ),
                'seed' : result['seed'],
                'mime_type': 'image/png',
                'nsfw': result['nsfw']
            })
            # imgBase64 = pil_to_b64( result['image'].convert( 'RGB' ) )
            #  base64.b64decode(base64text)
        output_data[ 'images' ] = images        
    except RuntimeError as e:
        output_data[ 'status' ] = 'failure'
        output_data[ 'message' ] = 'A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs for more details.'
        print(str(e))
    # return flask.jsonify( output_data )
    return StreamingResponse(io.BytesIO(base64.b64decode(output_data['images'][0]['base64'])), media_type='image/png')
