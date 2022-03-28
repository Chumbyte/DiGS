# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import matplotlib.animation as animation

output_path = '/home/chamin/PhD/DIBS/vis/reg_gt'
gt_reg_path = '/home/chamin/PhD/DIBS/DFaust/scripts'
pids = ['50002', '50004', '50007', '50009', '50020', '50021', '50022', '50025', '50026', '50027']
for pid in pids[2:]:
    human_dir = os.path.join(gt_reg_path, pid)
    vis_human_dir = os.path.join(output_path, pid)
    if not os.path.exists(vis_human_dir):
        os.mkdir(vis_human_dir)
    seqs = os.listdir(human_dir)
    for seq in seqs[:]:
        print(pid, seq)
        seq_dir = os.path.join(human_dir, seq)
        vis_seq_dir = os.path.join(vis_human_dir, seq)
        ims = sorted(os.listdir(vis_seq_dir))
        frames = []
        fig = plt.figure(figsize=(8,8))
        fig.tight_layout()
        plt.axis('off')
        for img in ims[:]:
            array = plt.imread(os.path.join(vis_seq_dir, img))
            frames.append([plt.imshow(array, animated=True)])
        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=False)
        ani.save(os.path.join(output_path, '{}-{}-ani.mp4'.format(pid, seq)))
        # plt.show()
        plt.close()