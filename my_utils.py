import os
from pathlib import Path
import pdb
import cv2
import shutil
from ultralytics import YOLO
from PIL import Image
import json
import numpy as np
import pickle



def save_train_val_test_vid_fn(anno_train_fp, anno_test_fp, train_ratio=0.85):
    with open(anno_train_fp) as f:
        train_lines = f.readlines()
    with open(anno_test_fp) as f:
        test_lines = f.readlines()
    
    train_val_vid_fn = set([line.split(',')[0] for line in train_lines])
    test_vid_fn = set([line.split(',')[0] for line in test_lines])

    np.random.shuffle(train_val_vid_fn)
    train_vid_fn = train_val_vid_fn[:int(train_ratio*len(train_val_vid_fn))]
    val_vid_fn = train_val_vid_fn[int(train_ratio*len(train_val_vid_fn)):]

    print(f'train: {len(train_vid_fn)}, val: {len(val_vid_fn)}, test: {len(test_vid_fn)}')
    pdb.set_trace()

    with open('train_vid_fn.txt', 'w') as f:
        for fn in train_vid_fn:
            f.write(f'{fn}\n')
    
    with open('val_vid_fn.txt', 'w') as f:
        for fn in val_vid_fn:
            f.write(f'{fn}\n')

    with open('test_vid_fn.txt', 'w') as f:
        for fn in test_vid_fn:
            f.write(f'{fn}\n')



def get_train_val_test_vid_fn():
    ls_fn = []
    for split in ['Train', 'Val', 'Test']:
        split_fn = []
        anno_fp = f'/data3/users/tungtx2/hand_gesture/ipn_hand_data/annotation/Annot_{split}List.txt'
        with open(anno_fp) as f:
            lines = f.readlines()
        split_fn = list(set([line.split(',')[0] for line in lines]))
        ls_fn.append(split_fn)

    return ls_fn


def split_train_val_annotation():
    np.random.seed(42)

    with open('ipn_hand_data/annotation/Annot_TrainList.txt') as f:
        anno = f.readlines()
    
    ls_vid_fn = list(set([line.split(',')[0] for line in anno]))
    np.random.shuffle(ls_vid_fn)
    num_train = int(len(ls_vid_fn)*0.85)
    ls_train_fn = ls_vid_fn[:num_train]
    ls_val_fn = ls_vid_fn[num_train:]

    with open('ipn_hand_data/annotation/Annot_TrainList_new.txt', 'w') as f:
        cnt = 0
        for line in anno:
            if line.split(',')[0] in ls_train_fn:
                f.write(line)
                cnt += 1
        print(f'write {cnt} lines to Annot_TrainList_new.txt')
    
    with open('ipn_hand_data/annotation/Annot_ValList_new.txt', 'w') as f:
        cnt = 0
        for line in anno:
            if line.split(',')[0] in ls_val_fn:
                f.write(line)
                cnt += 1
        print(f'write {cnt} lines to Annot_ValList_new.txt')


def gen_data_for_yolo_segment(
    mask_dir,
    all_frame_dir,
    out_dir,
    mask_thresh=50
):
    # make dirs
    train_image_dir = os.path.join(out_dir, 'train', 'images')
    train_label_dir = os.path.join(out_dir, 'train', 'masks')
    val_image_dir = os.path.join(out_dir, 'val', 'images')
    val_label_dir = os.path.join(out_dir, 'val', 'masks')
    test_image_dir = os.path.join(out_dir, 'test', 'images')
    test_label_dir = os.path.join(out_dir, 'test', 'masks')
    for dir in [train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_image_dir, test_label_dir]:
        os.makedirs(dir, exist_ok=True)

    # get train, val, test vid_fn
    train_vid_fn, val_vid_fn, test_vid_fn = get_train_val_test_vid_fn()

    # iterate over all frames
    for vid_fn in os.listdir(all_frame_dir):
        vid_dir = os.path.join(all_frame_dir, vid_fn)
        if vid_fn in train_vid_fn:
            image_dir = train_image_dir
            label_dir = train_label_dir
        elif vid_fn in val_vid_fn:
            image_dir = val_image_dir
            label_dir = val_label_dir
        elif vid_fn in test_vid_fn:
            image_dir = test_image_dir
            label_dir = test_label_dir
        
        image_dir = os.path.join(image_dir, vid_fn)
        label_dir = os.path.join(label_dir, vid_fn)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        for frame_fp in Path(vid_dir).glob('*.jpg'):
            shutil.copy(str(frame_fp), image_dir)
            mask_fp = os.path.join(mask_dir, vid_fn, frame_fp.name)
            mask = cv2.imread(mask_fp)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(mask, thresh=mask_thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
            cv2.imwrite(os.path.join(label_dir, frame_fp.stem+'.png'), mask)
            print(f'Done {frame_fp.name}')



def infer_yolo(model_fp, frame_dir, conf):
    """
        model_fp: path to trained yolo model
        frame_dir: dir of all frames which is the output of split_data function
        conf: conf threshold
    """
    model = YOLO(model_fp)

    for frame_fp in Path(frame_dir).rglob('*.jpg'):
        res = model.predict(
            source=str(frame_fp),
            conf=conf,
            max_det=1
        )
        res = res[0]

        if res.boxes.data.shape[0] > 0:
            max_idx = np.argmax(res.boxes.data[:, 4].cpu().numpy())
            conf = float(res.boxes.data[max_idx, 4].cpu().numpy())
            pred_box = res.boxes.xywhn[max_idx].cpu().numpy().tolist()
            x, y, w, h = pred_box
        else:
            conf, x, y, w, h = 0, 0, 0, 0, 0

        txt_fp = Path(frame_fp).with_suffix('.txt')
        os.makedirs(os.path.dirname(txt_fp), exist_ok=True)
        with open(txt_fp, 'w') as f:
            f.write(f'{conf} {x} {y} {w} {h}\n')
        print(f'Done {frame_fp}')


def split_data(anno_fp, frame_dir, root_out_dir):
    """
        anno_fp: txt filepath annotation of train or test set
        frame_dir: dir of all frames: frame_dir / vid_fn / frame.jpg
        root_out_dir: dir to save train or test set, root_out_dir / split / gesture_cl / vid_fn+start_fr+end_fr / frame.jpg
    """
    idx2name = {
        1: 'no_gesture',
        2: 'point_1_finger',
        3: 'point_2_finger',
        4: 'click_1_finger',
        5: 'click_2_finger',
        6: 'throw_up',
        7: 'throw_down',
        8: 'throw_left',
        9: 'throw_right',
        10: 'open_twice',
        11: 'double_click_1_finger',
        12: 'double_click_2_finger',
        13: 'zoom_in',
        14: 'zoom_out',
    }

    with open(anno_fp) as f:
        annos = f.readlines()
    
    anno_dict = {}
    for line in annos:
        vid_fn, gesture_code, gesture_cl, start_fr, end_fr, num_fr = line.strip().split(',')
        gesture_cl, start_fr, end_fr, num_fr = int(gesture_cl), int(start_fr), int(end_fr), int(num_fr)
        if vid_fn in anno_dict:
            anno_dict[vid_fn].append({
                'start_fr': start_fr,
                'end_fr': end_fr,
                'gesture_cl': gesture_cl,
            })
        else:
            anno_dict[vid_fn] = [{
                'start_fr': start_fr,
                'end_fr': end_fr,
                'gesture_cl': gesture_cl,
            }]
    
    # pdb.set_trace()
    for vid_fn, anno_data in anno_dict.items():
        vid_fr_dir = os.path.join(frame_dir, vid_fn)
        for data in anno_data:
            start_fr, end_fr, gesture_cl = data['start_fr'], data['end_fr'], data['gesture_cl']
            gesture_name = idx2name[gesture_cl]
            out_dir = os.path.join(root_out_dir, str(gesture_name), f'{vid_fn}_{start_fr}_{end_fr}')
            os.makedirs(out_dir, exist_ok=True)
            for fr in range(start_fr, end_fr+1):
                shutil.copy(os.path.join(vid_fr_dir, f'{vid_fn}_{fr:06d}.jpg'), out_dir)
            print(f'Done {vid_fn} - {start_fr} - {end_fr} - {gesture_cl}')


def count_images(dir):
    img_paths = list(Path(dir).rglob('*.jpg'))
    print(len(img_paths))



if __name__ == '__main__':
    from sys import argv
    # split_train_val_annotation()

    # split_data(
    #     anno_fp='/data3/users/tungtx2/hand_gesture/ipn_hand_data/annotation/Annot_TestList.txt',
    #     frame_dir='/data3/users/tungtx2/hand_gesture/ipn_hand_data/frames',
    #     root_out_dir='/data3/users/tungtx2/hand_gesture/classification_data/test'
    # )

    # for split in ['train', 'val', 'test']:
    #     infer_yolo(
    #         model_fp='/data3/users/tungtx2/hand_gesture/hand_detect/runs/detect/train3/weights/best.pt',
    #         frame_dir=f'/data3/users/tungtx2/hand_gesture/classification_data/{split}',
    #         conf=0.3,
    #     )

    # gen_data_for_yolo_segment(
    #     mask_dir='/data3/users/tungtx2/hand_gesture/original_segment_mask',
    #     all_frame_dir='/data3/users/tungtx2/hand_gesture/ipn_hand_data/frames',
    #     out_dir='/data3/users/tungtx2/hand_gesture/segment_data',
    #     mask_thresh=50
    # )

    count_images('/data3/users/tungtx2/hand_gesture/segment_data/train/images')