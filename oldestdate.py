import glob
import os
basedir='f:/ottawa_image_db/'
dirlist = os.listdir(basedir)

with open("f:/models/ff.csv", "w") as f:
    for d in dirlist:
        images = glob.glob(basedir+d+ '/*.jpg')
        if len(images)!=0:
            dlist=[]
            for image in images:
                fn=os.path.basename(image)
                dlist.append(fn[0:4])

            f.write("{},{}\n".format(d, min(dlist)))
            print(dlist,min(dlist))

