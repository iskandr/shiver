import math 
import os
import shutil
import sys
import tempfile

from subprocess import check_output, check_call  

import numpy as np 
from PIL import Image 
import progressbar 

def ensure_ffmpeg():
  devnull = open('/dev/null', 'w')
  try:
    check_call(['ffmpeg', '--help'], stderr = devnull, stdout = devnull)
  except:
    assert False, "Can't find ffmpeg"

def image_from_array(image):
  if isinstance(image, np.ndarray):
    image = Image.fromarray( (256 * image / image.max()).astype('uint8'))
  assert Image.isImageType(image)
  return image 

def resize(img, target_size = 300):
  """
  Resize so at least that the smallest dimension matches 
  the desired target size
  """
  img = image_from_array(img)
  width, height = img.size
  ratio = max( float(target_size) / width, float(target_size) / height) 
  return img.resize( (int(width * ratio), int(height * ratio)), Image.ANTIALIAS)

def images_to_movie(images, movie_name = 'movie.mpg', overwrite = True, 
                    n_interp_frames = 8,
                    frame_rate = 30, 
                    bitrate = '3000k'):
  """
  Takes a list of images, writes them to disk, uses ffmpeg to
  create a movie 
  """
  ensure_ffmpeg()
  # wrap directory creation so files get deleted afterward 
  try:
    base = tempfile.mkdtemp()

    n = len(images)
    total = 1 + (n-1)*(n_interp_frames+1)
    n_digits = int(math.ceil(np.log10(total)))
    format_string = "img%%0%dd.png" % n_digits
    print "Writing %d images..." % total 
    progress = progressbar.ProgressBar(maxval=total).start()
    last_image = None 
    def write_image(image):
      i = progress.currval
      filename = os.path.join(base, format_string % i)
      f = open(filename, 'w')
      image = image_from_array(image) 
      image.save(f, format='PNG')
      progress.update(i+1)
   
    for image in images:
      if last_image is not None: 
        # interpolate between successive images 
        for j in xrange(n_interp_frames):
          weight = float(j) / (n_interp_frames+1)
          new_image = last_image * (1-weight) + weight * image
          write_image(new_image)
      write_image(image)
      last_image = image  
    progress.finish()
    movie_cmd = ["ffmpeg",
                 '-an', # no sound! 
                 '-r',  '%d' % frame_rate, 
                 '-i', os.path.join(base, format_string), 
                 '-y' if overwrite else '-n', 
                 #'-vcodec', codec,
                 '-b:v', bitrate, 
                 movie_name]
    check_call(movie_cmd)
  finally:
    try:
        shutil.rmtree(base) # delete directory
    except OSError, e:
        if e.errno != 2: # code 2 - no such file or directory
            raise  


import scipy.ndimage 
from scipy.ndimage.morphology import *
def mk_movie(image_name='tyler.png', image_size = 200, n_frames=150, n_interp_frames=0):
  image = Image.open('tyler.png')
  image = resize(image, image_size)
  image = np.array(image)
  print image.shape
  image = image.astype('float') / image.max()
  nrows, ncols = image.shape[:2]
  images = [image]
  print "Generating frames..."
  x_grad_size = 3+np.random.randint(low=0,high=3,size=2)*2 
  y_grad_size = 3+np.random.randint(low=0,high=3,size=2)*2
  structure_size = 3+np.random.randint(low=0,high=6,size=2)*2
  structure = np.random.randn(structure_size[0], structure_size[1]) > np.random.randn()
  def dilate(x):
    return grey_dilation(x, size = structure_size, footprint = structure)
  def erode(x):
    return grey_erosion(x, size = structure_size, footprint = structure)
  progress = progressbar.ProgressBar()
  alpha = np.random.randn() * 0.1
  beta = np.random.rand() * 0.0
  original_g = image[:,:,1]
  for i in progress(xrange(n_frames)):
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    gray = 0.21 * r + 0.71 * g + 0.07 * b 
    x_grad = morphological_gradient(gray, x_grad_size, mode = 'nearest')
    y_grad = morphological_gradient(gray, y_grad_size, mode = 'nearest')
    hf = (x_grad + y_grad)**2 
    hf -= hf.min()
    hf /= hf.max()
    er = erode(r)
    dr = dilate(r)
    eg = erode(g)
    dg = dilate(g)
    eb = erode(b)
    db = dilate(b)
    r = r * hf + (1-hf)*(dr -er) + (1-hf)  *  alpha * (1-hf)**2 * original_g
    b = b * hf + (1-hf)*(db - eb) +(1-hf)  *  beta * (1-hf)**2 * original_g
    g = g * hf + (1-hf)  * (dg - eg)
    image = np.dstack([r,b,g])
    
    images.append(image)   
  images_to_movie(images, n_interp_frames=n_interp_frames)
  
if __name__ == '__main__':
  mk_movie('tyler.png', n_frames=100, n_interp_frames=7)
