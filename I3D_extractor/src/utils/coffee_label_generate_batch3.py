#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import pandas as pd
import os
import numpy as np
from pprint import pprint
import editdistance

def strip_name(name):
    name = name.strip('. \n')
    if name[0].isnumeric():
        name = ' '.join(name.split(' ')[1:])
    return name.lower()

def second2time(secs):
    minu = int(secs // 60)
    secs = secs % 60
    return f"{minu:02d}:{secs:02.2f}"
    
def second2frame(sec, fps=15):
    s = int(sec * fps)
    return s
    
class Segment():
    
    def __init__(self, name, start, end, ec=None, note=None):
        if end != -1:
            assert start < end, (name, start, end)
        self.name = name
        self.start = start
        self.end = end
        self.len = end - start
        self.ec = ec
        self.note = note
        
        
    def __str__(self):
        return f"{self.start}, {self.end} | {self.name} | {self.ec}"
    
    def __repr__(self):
        return str(self)

def parse_excel(excel):
    segments = []
    missing = []
    narration_level = None
    reached_end = False
    for i, row in excel.iterrows():
        if isinstance(row['action'], float):
            continue 
        
        action = strip_name(row['action'])
        if action == 'start':
            start_time = row['start time']
            continue
        if action == 'end':
            end_time = row['start time']
            reached_end = True
            continue
        if 'narration' in action.lower():
            narration_level = row['start time']
            continue
        

        start = row['start time']
        end = row['end time']        
        note = row['note']
        ec = None 
        if not isinstance(note, float):
            if note[0] == 'E':
                assert note[1].isnumeric()
                assert note[2] == ':' or note[2] == '.'
                ec = ( 'Error', int(note[1]) )
            elif note.lower().startswith('correction'):
                ec = ( "Correction", int(note[10]) )
                
        if not reached_end:
            seg = Segment(row['action'], start, end, ec=ec, note=note)
            segments.append(seg)
        else:
            seg = Segment(row['action'], 0.0, 0.1, ec=ec, note=note)
            missing.append(seg)
    
    segments.sort(key=lambda x: x.start)
    
    return segments, missing, narration_level, (start_time, end_time)

def generate_label(segments, start_time, end_time):
    label_segs = []
    if start_time < segments[0].start:
        label_segs = [ Segment('0.1. Start', start_time, segments[0].start) ]
    
    for i, seg in enumerate(segments):
        if len(label_segs) == 0:
            label_segs.append(seg)
        elif label_segs[-1].end <= seg.start:
            label_segs.append(seg)
        else:
            print('label_segs:', label_segs)
            ongoing = label_segs.pop()
            print('name:', ongoing.name)
            assert ongoing.name[0].isnumeric(), (ongoing, seg)
            assert seg.end <= ongoing.end
            assert seg.start >= ongoing.start
                
            if ongoing.start < seg.start:
                label_segs.append(Segment(ongoing.name, ongoing.start, seg.start, ec=ongoing.ec, note=ongoing.note))
            
            label_segs.append(seg)
                
            if ongoing.end > seg.end:
                label_segs.append(Segment(ongoing.name, seg.end, ongoing.end, ec=ongoing.ec, note=ongoing.note))
                
    
    if end_time == -1 or end_time > label_segs[-1].end:
        label_segs.append(Segment('0.2. End', label_segs[-1].end, end_time))
    
    return label_segs

action_list = [
 '0.0. BG',
 '0.1. Start',
 '0.2. End',
 '0.3. Clean/Reset',
 '0.4. Hesitate/Think',
 '0.5. Move Items to start the next step',
 '1.1. Measure 12 ounces of cold water',
 '1.2. Transfer water to a kettle',
 '1.3. Turn on the kettle to boil the water',
 '10. Discard the paper filter and coffee grounds',
 '11. Hold the cup of coffee in front of you',
 '2. Put dripper on mug',
 '3.1. Fold the paper filter in half to create a semi-circle',
 '3.2. Fold it in half again to create a quarter-circle',
 '4. Place the paper filter in the dripper and spread it into a cone',
 '5.0. Pour coffee beans from the bag',
 '5.1. Weigh 25 grams of coffee beans',
 '5.2. Put coffee beans in the coffee grinder',
 '5.3. Grind Coffee for 20 sec',
 '6. Transfer the grounds to the filter cone',
 '7.1 Wait for water to boil or cool down',
 '7.2. Check water temperature',
 '8.1. Pour a small amount of water on the grounds',
 '8.2. Wait for coffee to blossom',
 '9.1 Slowly pour the rest of the water in a circular motion',
 '9.2 Wait for water to drain']

FPS=10

#####################
# label2index mapping
L= action_list[:]
L.pop(1) # remove 0.1 start
L.pop(1) # remove 0.2 end
with open('../../dataset/PTG_coffee/label_v1/mapping.txt', 'w') as fp:
    for i, a in enumerate(L):
        fp.write(f'{i}|{a}\n')


###########################
# parse excel into action segments
xl = pd.ExcelFile('/mnt/raptor/datasets/PTG_dataset/recording/batch3/SBU_Data_Annotation_copy.xlsx')
xl = pd.ExcelFile('/mnt/raptor/datasets/PTG_dataset/recording/batch3/SBU_data_annotation_copy.xlsx')
labels = {}
for sheet_name in xl.sheet_names[0:]:
    print(sheet_name)
    excel = xl.parse(sheet_name) 
    segs, miss, narr, (start, end) = parse_excel(excel)
    print('narr:', narr)
    assert not isinstance(narr, float)

    # if len(miss) > 0:
    #     pprint(miss)
        
    label = generate_label(segs, start, end)
    for i, s in enumerate(label[1:]):
        assert s.start == label[i].end, (label[i], s)
    
    labels[sheet_name] = label


####################
# correct action names
actions = []
for vname, label in labels.items():
    for l in label:
        if l.name[0].isnumeric():
            actions.append(l.name)

taskActionConverse = {}
actions = list(set(actions))
for a in actions:
    seq = [ r for r in action_list if a[0] == r[0] ]
    dist = [ editdistance.eval(a.lower(), r.lower()) for r in seq ]
    match = np.argmin(dist)
    taskActionConverse[a] = seq[match]


for vname, label in labels.items():
    for l in label:
        if l.name[0].isnumeric():
            assert l.name in taskActionConverse
            l.name = taskActionConverse[l.name]



##########################
# collect the number of frames for each video
nframes = {}
for sheet_name in xl.sheet_names[0:]:    
    ffolder = f'/mnt/raptor/datasets/PTG_dataset/recording/batch3/frames_fps{FPS}/{sheet_name}/'
    images = os.listdir(ffolder)
    nf = max(map(lambda x: int(x.split('.')[0]), images))
    nframes[sheet_name] = nf

# for vname, label in labels.items():
#     with open(f"../../dataset/PTG_coffee/label_v1/annotations/{vname}.txt", "w") as fp:
#         for l in label:
#             if l.ec is None:
#                 ec = 'None'
#             else:
#                 ec = l.ec[0] + str(l.ec[1])
#             fp.write(f"{l.name}|{l.start}|{l.end}|{ec}\n")


########################
# generate framewise groundtruth labels
names = []
for vname, label in labels.items():
    print(vname)
    nf = nframes[vname]
    label_list = [ 'HIDDEN' ] * nf
    state_list = [ 'Normal' ] * nf # lapse label
    
    for l in label:
        
        print('nf:', nf)
        print('l:', l)
        start_frame = second2frame(l.start, fps=FPS)
        
        if l.name == '0.2. End' and l.end == -1:
            end_frame = nf
        else:
            end_frame = min(nf, second2frame(l.end, fps=FPS))        
    
        if l.name not in action_list or l.name in ['0.1. Start', '0.2. End']:
            name = '0.0. BG' # merge all task-irrelevant actions into BG class
        else:
            name = l.name
        
        print('start_frame, end_frame', start_frame, end_frame)
        assert 0 <= start_frame < end_frame <= nf, (l.name, start_frame, end_frame)
        names.append(name)
        for i in range(start_frame, end_frame):
            label_list[i] = name

        # TODO: need to relabel lapse error    
        # current: if an action contains lapse error, label all its frames as lapse error.
        if l.ec is not None and l.ec[0] == 'Error':
            for i in range(start_frame, end_frame):
                state_list[i] = "Error" 
            
    start = second2frame(label[0].start, fps=FPS)
    end = second2frame(label[-1].end, fps=FPS) if label[-1].end != -1 else nf
    
    assert 'HIDDEN' not in label_list[start:end]
            
    #with open(f"../../dataset/PTG_coffee/label_v1/groundTruth_10fps/{vname}.txt", "w") as fp:
    with open(f"/mnt/raptor/datasets/PTG_dataset/recording/batch3/processed/label_v1/groundTruth_fps10/{vname}.txt", "w") as fp:
        for l, s in zip(label_list, state_list):
            fp.write(f"{l}|{s}\n")

