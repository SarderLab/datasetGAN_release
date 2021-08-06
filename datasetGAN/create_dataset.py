import numpy as np
from imageio import imread, imwrite
import json, os
from glob import glob

path_to_data = '/hdd/brendonl/stylegan/results/00007-generate-images'
path_to_experiment = './experiments/glom_2.json'
path_to_masks = './dataset_release/annotation/training_data/glom_processed'


opts = json.load(open(path_to_experiment, 'r'))

# create latent array
ims = open('./dataset_release/annotation/training_data/glom_processed/images_used.txt','r').read().split('\n')
latents = []
for im in ims:
    latents.append(np.load('{}/latent_stylegan1_{}.npy'.format(path_to_data, im)))
latents = np.concatenate(latents,0)

latent_dir = os.path.dirname(opts['annotation_image_latent_path'])
if not os.path.exists(latent_dir):
    os.mkdir(latent_dir)

np.save(opts['annotation_image_latent_path'], latents)

# save avg latents
avg_latents = np.load('{}/avg_latent.npy'.format(path_to_data))
np.save(opts['average_latent'], avg_latents)

# create npy masks for annotated images
# masks = sorted(glob('{}/masks/*.png'.format(path_to_masks)))
# print(masks)

for i in range(len(ims)):
    mask_path = '{}/masks/image_{}.png'.format(path_to_masks,i)
    print('working on: '+mask_path)
    mask = imread(mask_path)[:,:,0]
    u, indices = np.unique(mask, return_inverse=True)
    int_mask = indices.reshape(mask.shape)
    np.save('{}/image_mask{}.npy'.format(path_to_masks,i), int_mask)

    im_path = '{}/image_{}.png'.format(path_to_masks,i)
    im = imread(im_path)
    imwrite('{}.jpg'.format(os.path.splitext(im_path)[0]),im)
