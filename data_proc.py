import re
import numpy as np
with open('SISSO.out', 'r') as file:
    output = file.read()
    file.close()
indices = [m.start() for m in re.finditer('LS_MaxAE', output)]
file_write = open('../RMSEs_train', 'a')
for i in indices:
    file_write.write(output[i+12:i+20]+'\n')
file_write.close()

with open('predict_Y.out', 'r') as file:
    output_val = file.read()
    file.close()
indices_val = [m.start() for m in re.finditer('MaxAE', output_val)]
file_write = open('../RMSEs_val', 'a')
for j in indices_val:
    file_write.write(output_val[j+15:j+27]+'\n')
file_write.close()

