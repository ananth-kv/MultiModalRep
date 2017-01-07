import os

## image and sentence parameters
input_json = '-input_json coco/test_cocotalk.json'
input_h5 = '-input_h5 coco/test_cocotalk.h5'
max_iters = '-max_iters 500000'
checkpoint_path = '-checkpoint_path checkpoints_now/'

#i_id = '-id 11'
logFile = '2>&1 | tee checkpoints_now/adv.log'

#gpu = '-gpuid 5'


cmd = ' '.join(c for c in ['th', 'train_adv.lua', input_h5, input_json, max_iters, logFile])

## th train_adv.lua -input_json coco/test_cocotalk.json -input_h5 coco/test_cocotalk.json -max_iters 500000


## Run the command
os.system(cmd)
        
print('Training Done....')
