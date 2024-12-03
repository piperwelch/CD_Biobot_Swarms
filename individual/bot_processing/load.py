'''
opens a body and cilia binvox file
converts to both to binary numpy arrays
performs post processing steps on the arrays
returns post-processed binary body and cilia numpy arrays
'''

import os
import sys
import numpy as np
import pickle
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot_processing import binvox_rw
from bot_processing.array_processing_functions import remove_internal_cilia, remove_noise, make_one_shape_only, shift_down, remove_protruding_voxels, orient, hollow, move_cilia_to_surface, shave, expose_cilia, spline_interp, map_cilia_to_body

DIR = os.path.abspath(__file__ + "/../../") + "/"  # absolute path of this file

def downscale(array, scale):
    return spline_interp(array, scale, order=0)
    
def process(body, hollow_out=False):
    body = expose_cilia(body)
    body = remove_protruding_voxels(body)
    body = orient(body)

    # Either hollow out the bot or remove the internal ciliated cells
    if hollow_out:
        body = hollow(body)
    else:
        body = remove_internal_cilia(body)
        # pass

    return body

def load_and_save_high_res(bot_name, body_fn, cilia_fn, hollow_out=False, bot_type=''):
    print("Loading full resolution bot...")
    # load body as array
    with open(DIR+body_fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        body = model.data.astype(int)  

    # load cilia as array
    with open(DIR+cilia_fn, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f) 
        cilia = remove_noise(model.data,MIN_CILIA_CC).astype(int) # remove some noise

    save_dir = DIR + "pickle/{}/{}".format(bot_type, bot_name)
    os.makedirs(save_dir, exist_ok=True)

    if hollow_out:
        file_name = '{}_fullRes_hollow.p'.format(bot_name)
    else:
        file_name = '{}_fullRes.p'.format(bot_name)
    
    save_fn = save_dir + '/{}'.format(file_name)

    with open(save_fn, 'wb') as f:
        pickle.dump([body,cilia], f)
        # pickle.dump(body, f)
    print('done.')
    return body, cilia, save_fn

def load(bot_name, full_res_bot_fn, scale, hollow_out=False, save_pickle=False, save_filename=None, bot_type=''):

    with open(DIR+'/'+full_res_bot_fn, 'rb') as f:
        body, cilia = pickle.load(f)
        
    print('Downscaling bot...')

    body = downscale(body, scale).astype(int)
    cilia = downscale(cilia, scale).astype(int)
    
    if REMAP:
        body_copy = body.copy()
        body_copy[cilia==1]=2
        body = map_cilia_to_body(body_copy, body)
        body_mask = make_one_shape_only(body)
        body[body_mask==False]=0
    elif CONNECT_CILIA:
        body_with_disconnected_cilia = body.copy() # save a copy of the body array without cilia to use as the true shape of the bot
        cilia_disconnected = cilia.copy()

        # # print("Combining arrays...")
        body[cilia==1] = 2 # set the material id for the ciliated cells to 2
        body_mask = make_one_shape_only(body)
        
        # Dealing with any disconnected cilia
        cilia_disconnected[body_mask]=0 
        if np.sum(cilia_disconnected)!=0:
            body_with_disconnected_cilia[cilia_disconnected==1]=2
            body = map_cilia_to_body(body_with_disconnected_cilia, body)

        body[body_mask==False]=0
    
    else:
        body[cilia==1] = 2 # set the material id for the ciliated cells to 2
        body_mask = make_one_shape_only(body)
        body[body_mask==False]=0

    # print("Additional processing...")
    body = process(body, hollow_out)

    body = shift_down(body)

    print("done.")

    # print("Resolution:", body.shape[0])
    # print('Voxel count:', np.sum(body>0))

    if save_pickle:

        save_dir = DIR + "pickle/{}/{}".format(bot_type, bot_name)
        os.makedirs(save_dir, exist_ok=True)

        if save_filename is not None:
            file_name = save_filename
        else:
            if hollow_out:
                file_name = '{}_res{}_hollow.p'.format(bot_name, body.shape[0])
            else:
                file_name = '{}_res{}.p'.format(bot_name, body.shape[0])
            
        save_fn = save_dir + '/{}'.format(file_name)

        with open(save_fn, 'wb') as f:
            pickle.dump([body,scale], f)
    
        return body, save_fn

    return body

if __name__=="__main__":

    global MIN_CILIA_CC, REMAP

    parser = argparse.ArgumentParser(description="Process some bots.")
    parser.add_argument("--full", dest="process_full", action='store_true')
    parser.add_argument("--linear", dest="linear", action='store_true', default=False)
    parser.add_argument("--circular", dest="circular", action='store_true', default=False)
    parser.add_argument("--res", dest="res", type=int, default=64)
    parser.add_argument("--bot", dest="bot_id")
    parser.add_argument("--min_cilia_size", dest="min_cilia", type=int, default=1100)
    parser.add_argument("--alpha", dest="alpha", type=int, default=12)
    parser.add_argument("--remap", dest="remap", action='store_true')
    parser.add_argument("--connect_cilia", dest="connect_cilia", action='store_true')

    args = parser.parse_args()

    if not args.linear and not args.circular:
        print("Specify whether the bot is linear or circular.")
        exit()
    if args.linear:
        bot_type='linear'
    if args.circular:
        bot_type = 'circular'

    BOT_ID = args.bot_id
    SCALE = args.res/256
    MIN_CILIA_CC = args.min_cilia
    REMAP = args.remap
    CONNECT_CILIA = args.connect_cilia

    if BOT_ID is None:
        sys.exit("Specify a bot ID using the --bot flag.")

    if args.process_full:

        # STEP 1: save the full resolution UNPROCESSED bot
        BODY_FILENAME = "binvox/"+BOT_ID+"/body_tform_alpha{}_pts450000.binvox".format(args.alpha)
        CILIA_FILENAME = "binvox/"+BOT_ID+"/cilia_tform.binvox"

        load_and_save_high_res(BOT_ID, BODY_FILENAME, CILIA_FILENAME, bot_type=bot_type)

    # STEP 2: DOWNSCALING
    FULL_RES_FILENAME = "pickle/{}/".format(bot_type)+BOT_ID+"/"+BOT_ID+"_fullRes.p"

    if not os.path.isfile(FULL_RES_FILENAME):
        sys.exit("Must process full resolution bot first. Run load.py with flag --full.")

    body, save_fn = load(BOT_ID, FULL_RES_FILENAME, SCALE, save_pickle=True, bot_type=bot_type)

    print("Voxel Count:", np.sum(body>0))
    print("Bot saved in", save_fn)
