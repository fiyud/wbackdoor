import os
import numpy as np
from torch.utils.data import Dataset


def _read_list(path):
    with open(path) as f:
        return [ln.strip().split()[0] for ln in f if ln.strip()]


class PersonInWiFi3D(Dataset):
    def __init__(self, split, data_root, experiment_name='one-person', num_person=1):
        self.split = split
        self.num_person = num_person
        sub = 'train_data' if split == 'training' else 'test_data'
        self.root = os.path.normpath(os.path.join(data_root, sub))
        lst = os.path.join(self.root, f'{sub}_list.txt')
        names = _read_list(lst)
        self.items = []
        for nm in names:
            try:
                pc = int(nm.split('_')[0][2])
            except (IndexError, ValueError):
                pc = 1
            keep = ({'one-person': 1, 'two-person': 2, 'three-person': 3}
                    .get(experiment_name, None))
            if keep is not None and pc != keep:
                continue
            self.items.append({
                'csi': os.path.normpath(os.path.join(self.root, 'csi_ap', nm + '.npy')),
                'kpt': os.path.normpath(os.path.join(self.root, 'keypoint', nm + '.npy')),
                'name': nm,
            })

    # ----- split the original read_frame into load + normalize -----------------------
    @staticmethod
    def load_raw(csi_path):
        return np.load(csi_path).astype(np.float32)

    @staticmethod
    def normalize(raw):
        amp = raw[:, :90, :]; ph = raw[:, 90:, :]
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-12)
        ph = (ph - ph.min()) / (ph.max() - ph.min() + 1e-12)
        return np.concatenate([amp, ph], axis=1).astype(np.float32)

    def load_pose(self, kpt_path):
        p = np.load(kpt_path).astype(np.float32)
        if p.ndim == 2:                                  # (14,3) -> (1,14,3)
            p = p[None]
        if p.shape[0] < self.num_person:                
            pad = np.zeros((self.num_person - p.shape[0],) + p.shape[1:], np.float32)
            p = np.concatenate([p, pad], 0)
        return p[:self.num_person]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        raw = self.load_raw(it['csi'])
        csi = self.normalize(raw)
        pose = self.load_pose(it['kpt'])
        return {'csi': csi, 'pose': pose, 'name': it['name']}
