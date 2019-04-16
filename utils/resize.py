import os
import glob
import cv2
import argparse
import multiprocessing

args = None

def resize(filename):
    src_filename = os.path.join(args.src_dir, filename)
    tgt_filename = os.path.join(args.tgt_dir, filename)
    #print(src_filename, tgt_filename)

    tgt_dir = os.path.join(args.tgt_dir, os.path.dirname(filename))
    #print(tgt_dir)
    if not os.path.exists(tgt_dir):
        print('making dir: ', tgt_dir)
        os.makedirs(tgt_dir, exist_ok=True)

    try:
        img = cv2.imread(src_filename)
        #print(img.shape)
        img = cv2.resize(img, (args.image_size, args.image_size))
        cv2.imwrite(tgt_filename, img)
    except:
        print('ERROR:', src_filename)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--tgt_dir', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()

    cwd = os.getcwd()
    os.chdir(args.src_dir)
    filenames = glob.glob('**/*.jpg', recursive=True)
    print(len(filenames))
    print(filenames[:10])
    os.chdir(cwd)

    pool = multiprocessing.Pool(processes=50)
    pool.map(resize, filenames)

    '''
    print(len(filenames))
    print(filenames[:10])
    print(os.path.dirname(filenames[0]))
    print(os.path.basename(filenames[0]))
    print(os.path.join(args.tgt_dir, filenames[0]))
    '''