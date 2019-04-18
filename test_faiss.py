# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import faiss  
import time
import settings

def hello():
    d = 64                           # dimension
    nb = 100000                      # database size
    nq = 10000                       # nb of queries
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    #print(xb[:2])
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

                    # make faiss available
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    print('xb.shape:', xb.shape)
    print('xq.shape:', xq.shape)
    ret = index.add(xb)                  # add vectors to the index
    print('ret:', ret)
    print(index.ntotal)

    k = 4                          # we want to see 4 nearest neighbors
    D, I = index.search(np.array([xb[150,:] + xb[550,:]]), 3) # sanity check
    print('I:', I)
    print('D:', D)
    D, I = index.search(xq[:2], k)     # actual search
    print(I[:50])                   # neighbors of the 5 first queries
    print(I[-5:])                  # neighbors of the 5 last queries

def save_vectors():
    d = 1024
    num_vectors = 1000000

    index = faiss.IndexFlatL2(d)   # build the index
    np.random.seed(1234)
    v = np.random.random((num_vectors, d)).astype('float32')
    index.add(v)
    v2 = np.random.random((num_vectors, d)).astype('float32')
    index.add(v2)

    faiss.write_index(index, os.path.join(settings.VECTOR_DIR, 'testvect.index'))

def load_search():
    index = faiss.read_index(os.path.join(settings.VECTOR_DIR, 'testvect.index'))
    np.random.seed(1234)
    q = np.random.random((1, 1024)).astype('float32')
    bg = time.time()
    D, I = index.search(q, 10)
    print('time:', time.time() - bg)
    print('D:', D)
    print('I:', I)

def save_ivf_vectors():
    d = 1024
    num_vectors = 1000000
    nlist = 200

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    np.random.seed(1234)
    v = np.random.random((num_vectors, d)).astype('float32')

    assert not index.is_trained
    index.train(v)
    assert index.is_trained
    index.add(v)

    #index = faiss.IndexFlatL2(d)   # build the index
    
    
    v2 = np.random.random((num_vectors, d)).astype('float32')
    index.train(v2)
    index.add(v2)

    faiss.write_index(index, os.path.join(settings.VECTOR_DIR, 'testvect_ivf.index'))

def load_search_ivf():
    index = faiss.read_index(os.path.join(settings.VECTOR_DIR, 'testvect_ivf.index'))
    np.random.seed(1234)
    q = np.random.random((1, 1024)).astype('float32')
    bg = time.time()
    D, I = index.search(q, 10)
    print('time:', time.time() - bg)
    print('D:', D)
    print('I:', I)

def load_search_ivf_2():
    print('loading')
    index = faiss.read_index(os.path.join(settings.VECTOR_DIR, 'FeatureNet_se_resnext50_32x4d.index'))
    np.random.seed(1234)
    q = np.random.random((1, 2048)).astype('float32')
    bg = time.time()
    print('searching...')
    D, I = index.search(q, 10)
    print('time:', time.time() - bg)
    print('D:', D)
    print('I:', I)

if __name__ == '__main__':
    #save_vectors()
    #load_search()
    #save_ivf_vectors()
    #load_search_ivf()
    load_search_ivf_2()
