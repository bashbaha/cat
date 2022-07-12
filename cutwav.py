import sys
import wave
import os
import numpy as np

#basepath = 'cat/data/business'
basepath = 'cat/data/librispeech'
f_wav =  os.path.join(basepath,sys.argv[1])
f_target = os.path.join(basepath,"pcm_vad_split_5.5sec",sys.argv[1][:-4])
data = wave.open(f_wav,'rb')

#assert data.getframerate() == 8000
dura_per = 5.5
dura_last_min = 3.0
frames_per = int(dura_per * data.getframerate())
frames_last_min = int(dura_last_min * data.getframerate())

segments = data.getnframes() // frames_per

pad_last_flag = False

if (data.getnframes() - segments * frames_per ) >= frames_last_min :
    pad_last_flag = True
    segments = segments + 1

idx = 0
while segments > 0:
    segments = segments -1
    begin_pos = idx * frames_per
    data.setpos(begin_pos)
    target_data = data.readframes(frames_per)
    targetfile = f_target+"."+str(idx)+".wav"
    if segments == 0 and pad_last_flag:
        target_data = target_data + target_data 
        print ('pad last: ' + targetfile )
    wave_write = wave.open(targetfile, 'wb')
    wave_write.setparams((data.getnchannels(),data.getsampwidth(),data.getframerate(),len(target_data), data.getcomptype(), data.getcompname()))
    wave_write.writeframes(target_data)
    wave_write.close()
    print('save: %s' % targetfile)
    idx = idx + 1

data.close()
