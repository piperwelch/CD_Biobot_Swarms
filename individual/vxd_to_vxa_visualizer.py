import os,sys
from viz_vxa_generator import vxa_from_vxd


vxd_file = sys.argv[1]
vxa_from_vxd(vxd_file)

os.system("voxcraft-viz test.vxa")
