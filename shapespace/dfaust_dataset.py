# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import torch
import torch.utils.data as data
import numpy as np
import os, json

class DFaustDataSet(data.Dataset):

    def __init__(self, dataset_path, split_path, gt_path=None, scan_path=None, points_batch=16384, d_in=3, with_normals=False,\
        local_sigma=0.01, global_sigma=1.2):
        # dataset_path, gt_path, scan_path

        self.local_sigma = local_sigma
        self.global_sigma = global_sigma

        print('getting npy files')
        dataset_path = os.path.expanduser(dataset_path); assert os.path.exists(dataset_path)
        split_path = os.path.expanduser(split_path); assert os.path.exists(split_path), split_path
        with open(split_path, "r") as f:
            split = json.load(f)
        self.npyfiles_mnfld = get_instance_filenames(dataset_path, split) # list of filenames
        print('getting normalisation files')
        self.normalization_files = get_instance_filenames(dataset_path, split, '_normalization') # list of filenames

        # Used only for evaluation
        if gt_path is not None:
            print('getting gt files')
            gt_path = os.path.expanduser(gt_path); assert os.path.exists(gt_path)
            self.gt_files = get_instance_filenames(os.path.expanduser(gt_path),split,'','obj')
            self.shapenames = [x.split('/')[-1].split('.obj')[0] for x in self.gt_files]
        if scan_path is not None:
            print('getting scan files')
            scan_path = os.path.expanduser(scan_path); assert os.path.exists(scan_path)
            self.scans_files = get_instance_filenames(os.path.expanduser(scan_path), split,'','ply')

        self.points_batch = points_batch
        self.with_normals = with_normals
        self.d_in = d_in

    def load_points(self, index):
        return np.load(self.npyfiles_mnfld[index]) # (250000, 6) which has xyz, normal xyz

    def get_info(self, index):
        shape_name, pose, tag = self.npyfiles_mnfld[index].split('/')[-3:]
        return shape_name, pose, tag[:tag.find('.npy')]

    def __getitem__(self, index):

        point_set_mnlfld = torch.from_numpy(self.load_points(index)).float() # (250000, 6) which has xyz, normal xyz

        random_idx = torch.randperm(point_set_mnlfld.shape[0])[:self.points_batch]
        point_set_mnlfld = torch.index_select(point_set_mnlfld, 0, random_idx) # (pnts, 6)
        
        mnfld_points = point_set_mnlfld[:, :self.d_in]

        if self.with_normals:
            normals = point_set_mnlfld[:, -self.d_in:]  # todo adjust to case when we get no sigmas
        else:
            normals = torch.empty(0)

        sample_size, dim = mnfld_points.shape # (16384, 3)

        # sample_local   = mnfld_points + (torch.randn_like(mnfld_points) * self.local_sigma)        
        # sample_global = (torch.rand(sample_size // 8, dim) * (self.global_sigma * 2)) - self.global_sigma # (2048,3)
        # nonmnfld_points = torch.cat([sample_local, sample_global], dim=-2) # (18432, 3)
        
        nonmnfld_points = (torch.rand(sample_size, dim) * (self.global_sigma * 2)) - self.global_sigma


        return mnfld_points, nonmnfld_points, normals, index

    def __len__(self):
        return len(self.npyfiles_mnfld)


def get_instance_filenames(base_dir, split, ext='', format='npy'):
    npyfiles = []
    l = 0
    for dataset in split:
        print(dataset)
        for class_name in split[dataset]:
            # print(class_name)
            for instance_name in split[dataset][class_name]:
                j = 0
                if split[dataset][class_name][instance_name] == "all":
                    pass
                    instance_dir = os.path.join(base_dir, class_name, instance_name)
                    shapes = ['.'.join(x.split('.')[:-1]) for x in os.listdir(instance_dir) if '_normalization' not in x]
                    for shape in shapes:
                        instance_filename = os.path.join(base_dir, class_name, instance_name,
                                                        shape + "{0}.{1}".format(ext, format))
                        l = l + 1
                        j = j + 1
                        npyfiles.append(instance_filename)
                else: # else assume list
                    for shape in split[dataset][class_name][instance_name]:

                        instance_filename = os.path.join(base_dir, class_name, instance_name,
                                                        shape + "{0}.{1}".format(ext, format))
                        if not os.path.isfile(instance_filename):
                            print(
                                'Requested non-existent file "' + instance_filename + "' {0} , {1}".format(l, j)
                            )
                            l = l + 1
                            j = j + 1
                        npyfiles.append(instance_filename)
    return npyfiles



if __name__ == "__main__":
    np.random.seed(0)
    dataset_path = '~/PhD/DIBS/dfaust_processed'
    split_path = '~/PhD/DIBS/DFaust/splits/dfaust/train_all.json'
    split_path = '~/PhD/DIBS/DFaust/splits/dfaust/test_all.json'
    # code_path = os.path.abspath(os.path.curdir)
    # split_path = os.path.join(code_path, 'splits', 'dfaust/train_all.json')
    # split_path = os.path.join(code_path, 'splits', 'dfaust/test_all.json')
    gt_path = '~/PhD/DIBS/DFaust/scripts'
    scan_path = '~/PhD/DIBS/DFaust/scans'
    # ds = DFaustDataSet(dataset_path, split_path, gt_path=gt_path, scan_path=scan_path, points_batch=16384, d_in=3, with_normals=False)
    ds = DFaustDataSet(dataset_path, split_path, gt_path=gt_path, scan_path=scan_path, points_batch=16384, d_in=3, with_normals=True)
    import pdb; pdb.set_trace()