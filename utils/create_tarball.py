import os
import multiprocessing
import subprocess

def create_tarball(index):
    print('task:', index)
    tarname = 'imgs_256_{:01x}_{:01x}.tar'.format(index // 16, index % 16)
    to_be_compressed = '{:01x}/{:01x}/*'.format(index //16, index % 16)
    cmd = 'tar cvf {} {}'.format(tarname, to_be_compressed)
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

if __name__ == '__main__':
    #sub_dirs = [ '{:01x}'.format(i // 16) + '/' + '{:01x}'.format(i % 16) for i in range(256)]
    indices = list(range(256))
    print(indices)

    pool = multiprocessing.Pool(processes=50)
    pool.map(create_tarball, indices)