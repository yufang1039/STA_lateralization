clear;
md_file = '/Users/wenwen/Downloads/STA/CurrentFlowSimulation/mdVol_84_18_9_18.mat';

mag_factor = 1.2;% right STA

STA_MNI_right = [18 9 18];

load(md_file,'mdVol');
load('mri2mni.mat','mri2mni');
mdVol = mdVol.*mag_factor;

cm = colormap(jet(256)); 
cm = [1 1 1;cm];
pos = [118 140 212];
clim = [0 0.24];

sliceshow(mdVol,pos,cm,clim,'(V/m)',[],[],mri2mni);

