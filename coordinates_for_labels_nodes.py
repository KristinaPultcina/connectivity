import mne
import pandas as pd
import numpy as np


rh_list = mne.read_labels_from_annot("fsaverage", parc = "HCPMMP1", hemi ="rh") 
rh_list.pop(0)
lh_list = mne.read_labels_from_annot("fsaverage", parc = "HCPMMP1", hemi = "lh")
lh_list.pop(0)
lh_list_coord = [lh_list.center_of_mass() for label in lh_list] 
lh_list_coord =tuple()
for l in lh_list:
    c= l.center_of_mass(subject="fsaverage")
    cord= mne.vertex_to_mni(c,0,subject="fsaverage")
    lh_list_coord = lh_list_coord + (cord,)

lh_list_names = [label.name for label in lh_list] 

my_df = pd.DataFrame(lh_list_coord, columns = ['x.mni','y.mni','z.mni'])
my_df['ROI.name']=lh_list_names
my_df.to_csv('/home/pultsinak/Рабочий стол/connectivity/lh_hcp_coord.csv')

rh_list_coord =tuple()
for r in rh_list:
    c= r.center_of_mass(subject="fsaverage")
    cord= mne.vertex_to_mni(c,1,subject="fsaverage")
    rh_list_coord = rh_list_coord + (cord,)

rh_list_names = [label.name for label in rh_list] 

my_df = pd.DataFrame(rh_list_coord, columns = ['x.mni','y.mni','z.mni'])
my_df['ROI.name']=rh_list_names
my_df.to_csv('/home/pultsinak/Рабочий стол/connectivity/rh_hcp_coord.csv')
