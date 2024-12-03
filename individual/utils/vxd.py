'''
writes to or generates a .vxd file
'''

import numpy as np
from lxml import etree

class VXD:

    def __init__(self):
        self.generate_vxd()

    def generate_vxd(self):
        root = etree.XML("<VXD></VXD>")
        self.tree = etree.ElementTree(root)

    def set_vxd_tags(self, body, cilia=None, record_history=False, record_voxels=False, force_field_on=False):
        root = self.tree.getroot()
        if record_history:
            history = etree.SubElement(root, "RecordHistory")
            history.set('replace', 'VXA.Simulator.RecordHistory')
            etree.SubElement(history, "RecordStepSize").text = '100'
            if record_voxels:
                etree.SubElement(history, "RecordVoxel").text = '1'
            else:
                etree.SubElement(history, "RecordVoxel").text = '0'
            etree.SubElement(history, "RecordLink").text = '0' 
            etree.SubElement(history, "RecordFixedVoxels").text = '0'  
            etree.SubElement(history, "RecordCoMTraceOfEachVoxelGroupfOfThisMaterial").text = '1'  # record CoM trace
        
        if force_field_on:
            # find height of bot
            z = body.shape[2]
            for i in range(body.shape[2]):
                if np.sum(body[:,:,i])==0:
                    z-=1

            # set force field
            force_field = etree.SubElement(root, "ForceField")
            force_field.set('replace', 'VXA.Simulator.ForceField')
            z_forcefield = etree.SubElement(force_field, "z_forcefield")
            mtMUL = etree.SubElement(z_forcefield, "mtMUL")
            etree.SubElement(mtMUL, "mtCONST").text = '-1'
            mtGREATERTHAN = etree.SubElement(mtMUL, 'mtGREATERTHAN')
            etree.SubElement(mtGREATERTHAN, "mtVAR").text = 'z'
            etree.SubElement(mtGREATERTHAN, "mtCONST").text = str(z*0.01)

        self.set_data(body, cilia)

    def set_data(self, body, cilia=None):
        root = self.tree.getroot()

        X_Voxels, Y_Voxels, Z_Voxels  = body.shape
        body_flatten = np.zeros((X_Voxels*Y_Voxels, Z_Voxels),dtype=np.int8)
        for i in range(Z_Voxels):
            body_flatten[:,i] = body[:,:,i].flatten()
        
        structure = etree.SubElement(root, "Structure")
        structure.set('replace', 'VXA.VXC.Structure')
        structure.set('Compression', 'ASCII_READABLE')

        etree.SubElement(structure, "X_Voxels").text = str(X_Voxels)
        etree.SubElement(structure, "Y_Voxels").text = str(Y_Voxels)
        etree.SubElement(structure, "Z_Voxels").text = str(Z_Voxels)

        # set body data
        Data = etree.SubElement(structure, "Data")
        for i in range(Z_Voxels):
            string = "".join([f"{c}" for c in body_flatten[:,i]])
            etree.SubElement(Data, "Layer").text = etree.CDATA(string)

        # set cilia forces
        BaseCiliaForce = etree.SubElement(structure, "BaseCiliaForce")
        if cilia is not None:
            cilia_flatten = np.zeros((X_Voxels*Y_Voxels*3, Z_Voxels))
            for i in range(Z_Voxels):
                cilia_flatten[:,i] = cilia[:,:,i,:].flatten()

            for i in range(Z_Voxels):
                string = ",".join([f"{c}" for c in cilia_flatten[:,i]])
                etree.SubElement(BaseCiliaForce, "Layer").text = etree.CDATA(string)

    # save the vxd to data folder
    def write(self, filename):
        with open('data/{}'.format(filename), 'w+') as f:
            f.write(etree.tostring(self.tree, encoding="unicode", pretty_print=True))