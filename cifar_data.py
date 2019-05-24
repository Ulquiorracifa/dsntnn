import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['eval'] = ['test_batch']
  return file_names

if __name__ == '__main__':
    file_path = 'D:\Development\pycharm_workspace\Keras\data\cifar-10-python\cifar-10-batches-py'
    filename_dic = _get_file_names()
    file_dic = {}
    for c in filename_dic['train']:
        file_dic = unpickle(os.path.join(file_path, c))
        print('filename:    ' + c + '   filedic:    '+ str(file_dic.keys()))
        #filename:    data_batch_1   filedic:    dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        img = file_dic[b'data']
        print(img.shape)
        break