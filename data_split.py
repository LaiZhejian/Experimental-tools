import numpy as np

file1 = 'train.he-en.he.bpe'
file2 = 'train.he-en.en.bpe'

# prefix = "/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-heen/"
# with open(prefix + file1) as f1, open(prefix + file2) as f2, \
#         open(prefix + "train.he-en.he", "w") as train_fs, open(prefix + "train.he-en.en", "w") as train_ft, \
#         open(prefix + "dev.he-en.he", "w") as dev_fs, open(prefix + "dev.he-en.en", "w") as dev_ft,\
#         open(prefix + "test.he-en.he", "w") as test_fs, open(prefix + "test.he-en.en", "w") as test_ft:
#     f1 = f1.readlines()
#     f2 = f2.readlines()
#
#     random_index = np.random.permutation(len(f1))
#     train_index = random_index[:3000000]
#     dev_index = random_index[3000000: 3000000 + 2000]
#     test_index = random_index[3000000 + 2000: 3000000 + 2000 + 2000]
#     for i in train_index:
#         train_fs.write(f1[i])
#         train_ft.write(f2[i])
#     for i in dev_index:
#         dev_fs.write(f1[i])
#         dev_ft.write(f2[i])
#     for i in test_index:
#         test_fs.write(f1[i])
#         test_ft.write(f2[i])

with open('/home/data_91_c/zhangy/record-221117/wmt20-ende-data-bpe-parallel/all.en-de.en') as f1, \
        open('/home/data_91_c/zhangy/record-221117/wmt20-ende-data-bpe-parallel/all.en-de.de') as f2, \
        open('/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/zhangyu/train.zh-en.bpe.en', 'w') as f3, \
        open('/home/data_91_c/laizj/data/WMT/23/parallel/wmt23-ende/zhangyu/train.zh-en.bpe.de', 'w') as f4:
    per = np.random.permutation(23440059)[:3500000]
    per = np.sort(per)
    count = 0
    for idx, (l1, l2) in enumerate(zip(f1, f2)):
        if idx == per[count]:
            count += 1
            f3.write(l1)
            f4.write(l2)
