import os
from os import listdir
from os.path import isfile, join

path = "./graphs/"
for c in range(2, 20):
    for d in range(2, 5):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        files = {}
        for file in onlyfiles:
            sfc, sfd, sfn = file.split("_")
            fc, fd, fn = [int(p.split("=")[1]) for p in [sfc, sfd, sfn]]
            if fc == c and fd == d:
                files[fn] = file
        max_nodes = max(list(files.keys()))
        del files[max_nodes]
        for file in files:
            os.remove(join(path, files[file]))
