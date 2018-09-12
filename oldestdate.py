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

#Oldest and youngest only
basedir='f:/ottawa_image_db/'
dirlist = os.listdir(basedir)

with open("f:/models/ff_old_young2.csv", "w") as f:
    f.write("LAT,LON,MINY,MAXY,DRANGE,NPHOTO,NYEARS\n")
    for d in dirlist:
        images = glob.glob(basedir+d+ '/*.jpg')
        if len(images)!=0:
            dlist=[]
            fnamelist=[]
            for image in images:
                fn=os.path.basename(image)
                fnamelist.append(fn)
                dlist.append(fn[0:4])
            mx=max(dlist)
            mn=min(dlist)
            diff=int(mx)-int(mn)
            ld=len(dlist)
            lds=len(set(dlist))
            if (((mx=='2016') | (mx=='2015') | (mx=='2014')) & (mn=='2007')):
                f.write("{},{},{},{},{},{}\n".format(d, mn,mx,diff,ld,lds))
            #print(dlist,min(dlist))


# All years for all images with coords
with open("f:/models/allyearvector_coords.csv", "w") as f:
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
                f.write("{},{}\n".format(d,i))


import shutil
#newdata
basedir='f:/ottawa_image_db/'
otherdir = 'f:/ottawaimage4/'

dirlist = os.listdir(basedir)
y1 = '2007'
y2 = '2009'

for d in dirlist:
    images = glob.glob(basedir+d+ '/*.jpg')
    if len(images)>1:

        dlist=[]
        fnamelist=[]
        for image in images:
            fn=os.path.basename(image)
            fnamelist.append(fn)
            dlist.append(fn[0:4])

        mx=max(dlist)
        mn=min(dlist)
        diff=int(mx)-int(mn)
        ld=len(dlist)
        lds=len(set(dlist))
        if (y1 in set(dlist)) & (y2 in set(dlist)):
            newpath = otherdir + os.sep + d
            os.makedirs(newpath)
            shutil.copy(images[dlist.index(y1)], newpath+os.sep+fnamelist[dlist.index(y1)])
            shutil.copy(images[dlist.index(y2)], newpath + os.sep + fnamelist[dlist.index(y2)])
            #f.write("{},{},{},{},{},{}\n".format(d, mn,mx,diff,ld,lds))