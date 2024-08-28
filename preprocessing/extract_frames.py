import os

task = 'pinwheels'

filenames = os.listdir('data/EgoPER/%s/trim_videos'%(task))

for filename in filenames:
    id = filename[:-4]
    if not os.path.exists('data/EgoPER/%s/frames_10fps/%s'%(task, id)):
        os.mkdir('data/EgoPER/%s/frames_10fps/%s'%(task, id))
    os.system('ffmpeg -i data/EgoPER/%s/    trim_videos/%s.mp4 '%(task, id) + '-vf "fps=10" data/EgoPER/%s/frames_10fps/%s'%(task, id) + '/%06d.png')