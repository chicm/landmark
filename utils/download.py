from multiprocessing import Pool
import subprocess


def download_file(file_name):
    cmd = './download.sh ' + file_name
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

def test(file_name):
    print(file_name)

num_workers = 20
if __name__ == '__main__':
    p = Pool(num_workers)
    file_names = ['images_{:03d}.tar'.format(i) for i in range(500)]
    p.map(download_file, file_names, num_workers)
    p.close()
    p.join()
    print('done')
