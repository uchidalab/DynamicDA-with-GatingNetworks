import os, cv2
from PIL import Image

def make_gif(path, tag):
    imgs = []
    
    for i in range(299):
        copied = Image.open(path + '/{}{}.png'.format(tag, i+1))
        imgs.append(copied.convert('P'))
        del copied
    
    imgs[-1].save('{}/{}_{}.gif'.format(path, os.path.basename(path), tag), save_all=True, append_images=imgs[1:], optimize=False, loop=0, duration=140)


if __name__ == '__main__':
    make_gif('../data/result_hist/ShapeletSim_identity_VS_noise', 'train')
    print('train finished.')
    make_gif('../data/result_hist/ShapeletSim_identity_VS_noise', 'test')
    print('test finished.')
