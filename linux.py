#密度图生成函数
def Den_Gen():
    impath = '/home/yyyin/project/multi/ShanghaiTech/part_A/train_data/images/IMG_'
    g_path = '/home/yyyin/project/multi/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 301)):
        n, tot, img = Loader_3(i, impath, g_path, )
        np.save(
            '/home/yyyin/project/multi/ShanghaiTech/part_A/train_data/density-map-8/DEN_IMG_' + str(i) + '.npy',
            tot)

    impath = '/home/yyyin/project/multi/ShanghaiTech/part_A/test_data/images/IMG_'
    g_path = '/home/yyyin/project/multi/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 183)):
        n, tot, img = Loader_3(i, impath, g_path)
        np.save(
            '/home/yyyin/project/multi/ShanghaiTech/part_A/test_data/density-map-8/DEN_IMG_' + str(i) + '.npy',
            tot)

    impath = '/home/yyyin/project/multi/ShanghaiTech/part_B/train_data/images/IMG_'
    g_path = '/home/yyyin/project/multi/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 401)):
        n, tot, img = Loader_3(i, impath, g_path)
        np.save(
            '/home/yyyin/project/multi/ShanghaiTech/part_B/train_data/density-map-8/DEN_IMG_' + str(i) + '.npy',
            tot)

    impath = '/home/yyyin/project/multi/ShanghaiTech/part_B/test_data/images/IMG_'
    g_path = '/home/yyyin/project/multi/ShanghaiTech/part_B/test_data/ground-truth/GT_IMG_'
    for i in tqdm(range(1, 317)):
        n, tot, img = Loader_3(i, impath, g_path)
        np.save(
            '/home/yyyin/project/multi/ShanghaiTech/part_B/test_data/density-map-8/DEN_IMG_' + str(i) + '.npy',
            tot)
