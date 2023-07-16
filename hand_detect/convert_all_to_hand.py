# convert all class to hand
from pathlib import Path
import os

for txt_fp in Path('/data3/users/tungtx2/hand_gesture/hand_detect/yolo_hagrid_30k_split').rglob('*.txt'):
  with open(txt_fp) as f:
    lines = f.readlines()

  new_lines = []
  for line in lines:
    x, y, w, h = line.strip().split()[1:]
    new_line = [0, x, y, w, h]
    new_lines.append(f'0 {x} {y} {w} {h}\n')

  with open(txt_fp, 'w') as f:
    for line in new_lines:
      f.write(line)
  print(f'Done {txt_fp}')
