import os
import settings

#DATA_DIR = '/data/retrieval'
DATA_DIR = '/mnt/chicm/data/retrieval'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_imgs_256_crop')
INDEX_IMG_DIR = os.path.join(DATA_DIR, 'index_imgs_256_crop')
TEST_IMG_DIR = os.path.join(settings.DATA_DIR, 'test', 'images_jpg_256')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
VECTOR_DIR = os.path.join(DATA_DIR, 'vectors')
