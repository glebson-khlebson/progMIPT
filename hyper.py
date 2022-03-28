with open('SISSO_predict_para', 'r') as file:
    STR = file.read()
    file.close()
nsample = str(45 - int(STR[0]))
with open('SISSO.in', 'r') as file_read:
    s  = file_read.read()
    s = s.replace('nsample=45', 'nsample='+nsample)
    file_read.close()
with open('SISSO.in', 'w') as file_write:
    file_write.write(s)
    file_write.close()
