import glob
import os
basedir='f:/ottawa_image_db/'
dirlist = os.listdir(basedir)

with open("f:/models/ffxx.csv", "w") as f:
    f.write("LAT,LON,MINY,MAXY,DRANGE,NPHOTO,NYEARS\n")
    for d in dirlist:
        images = glob.glob(basedir+d+ '/*.jpg')
        if len(images)!=0:
            dlist=[]
            for image in images:
                fn=os.path.basename(image)
                dlist.append(fn[0:4])
            mx=max(dlist)
            mn=min(dlist)
            diff=int(mx)-int(mn)
            ld=len(dlist)
            lds=len(set(dlist))
            f.write("{},{},{},{},{},{}\n".format(d, mn,mx,diff,ld,lds))
            #print(dlist,min(dlist))

# All years for all images
with open("f:/models/allyearvector.csv", "w") as f:
    #f.write("LAT,LON,MINY,MAXY,DRANGE,NPHOTO,NYEARS\n")
    for d in dirlist:
        images = glob.glob(basedir+d+ '/*.jpg')
        if len(images)!=0:
            dlist=[]
            for image in images:
                fn=os.path.basename(image)
                dlist.append(fn[0:4])
            mx=max(dlist)
            mn=min(dlist)
            diff=int(mx)-int(mn)
            ld=len(dlist)
            lds=len(set(dlist))
            for i in dlist:
                f.write("{}\n".format(i))