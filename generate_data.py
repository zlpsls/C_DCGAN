import os
from glob import glob

'''
这个是生成数据集list的文件，相当于把训练集和label的列表事先存在文件data.flist和label.flist里。
由于2、4、6类图片差不多只有其他三类图片的一半，我把它们在list里重复了两次~
'''
filefolder = ['cr', 'in', 'pa', 'ps', 'rs', 'sc']
filepath = './data/data.flist'
f1 = open(filepath, 'w+')
labelpath = './data/label.flist'
f2 = open(labelpath, 'w+')
for i in range(1, len(filefolder), 2):
    path = './data/' + filefolder[i]
    img_list = glob(os.path.join(path, '*.jpg'))
    for img_name in img_list:
        f1.write(img_name)
        f2.write('%d' % (i))
        f1.write('\n')
        f2.write('\n')
for i in range(len(filefolder)):
    path = './data/' + filefolder[i]
    img_list = glob(os.path.join(path, '*.jpg'))
    for img_name in img_list:
        f1.write(img_name)
        f2.write('%d' % (i))
        f1.write('\n')
        f2.write('\n')
