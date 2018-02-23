from pathlib import Path

def kitti_paths(root: Path):
    dir_l = root / 'image_2'
    dir_r = root / 'image_3'
    dir_d = root / 'disp_noc_0'
    paths = []
    for path in dir_d.iterdir():
        paths.append({
            'left': dir_l / path.name,
            'right': dir_r / path.name,
            'disparity': path,
        })
    return paths