import os
from PIL import Image
import json
#
# data = json.load(open('class_data.json', 'r'))
# for i in range(len(data)):
#     filename = data[i][0].split('/')[-1]
#     if 'goire' in filename and 'gre' not in filename:
#         pic = Image.open(data[i][0])
#         print(data[i][0])
#         new_name = data[i][0].split('/')[0] + '/zxzxzx' + filename[9::]
#         pic.save(new_name)
#         data[i][0] = new_name
#
# with open('删除奇特字符.json', 'w') as fp:
#     fp.write(json.dumps(data))


# arnold-
# data = json.load(open('删除奇特字符.json', 'r'))
# for i in range(len(data)):
#     filename = data[i][0].split('/')[-1]
#     if 'arnold-' in filename :
#         pic = Image.open(data[i][0])
#         new_name = data[i][0].split('/')[0] + '/arnold-backlin' + filename[15::]
#         print(new_name)
#         pic.save(new_name)
#         data[i][0] = new_name
#
# with open('删除奇特字符.json', 'w') as fp:
#     fp.write(json.dumps(data))



# pic = Image.open(filename)
#         new_name = "new-%d.jpg"%(count)
#         filename.split('/')[0] + '/' + new_name
#         count+=1
#         pic.save(new_name)
#         print(new_name)
#         data[i][0]

def is_caozao(filename):
    res = 1
    for char in filename.split('/')[-1]:
        if '0'<=char<='9':
            continue
        if 'a'<=char<='z':
            continue
        if 'A'<=char<='Z':
            continue
        if char in ['-' , '_' , '(' , ')' , "." , "'" , "-"] :
            continue
        res = 0
    return res

data = json.load(open('class_data.json', 'r'))
count = 0
for i in range(len(data)):
    filename = data[i][0]
    if is_caozao(filename)==0:
        pic = Image.open(filename)
        new_name = "aaaaaaaaaanew-%d.jpg" % (count)
        new_name = filename.split('/')[0] + '/' + new_name
        count += 1
        pic.save(new_name)
        print(new_name)
        data[i][0] = new_name

with open('删除奇特字符.json', 'w') as fp:
    fp.write(json.dumps(data))