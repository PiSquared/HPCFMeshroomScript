#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import time
from rps import RPS
import psutil

#initialize the robust photometric stereo solver
rps = RPS()

#take in the arguments
print(len(sys.argv))

if (len(sys.argv) != 5):
    print("usage: python run_rps.py <mask> <lights> <imgs> <output>")
    print("Must pass 5 arguments.")
    sys.exit(0)
mask = sys.argv[1]
lights = sys.argv[2]
imgs = sys.argv[3]
output = int(sys.argv[4])

print("Mask: %s" % mask)
print("Lights: %s" % lights)
print("Images: %s" % imgs)
print("Output file: %d" % output)

# Load data into rps
rps.load_mask(filename=mask)
rps.load_lightnpy(filename=lights)
rps.load_images(foldername=imgs, ext=png)
# start timer
start = time.time()
rps.solve(METHOD)
# finish timer
elapsed_time = time.time() - start
print("Photometric stereo: time to process:{0}".format(elapsed_time) + "[sec]")
#output normal map
rps.save_normalmap(filename=output)