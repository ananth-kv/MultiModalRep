grep -sir "errG: " checkpoints_now/adv.log | awk -F " " '{print $5}' > checkpoints_now/errG.csv

grep -sir "errD: " checkpoints_now/adv.log | awk -F " " '{print $7}' > checkpoints_now/errD.csv

