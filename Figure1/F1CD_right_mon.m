close all
clc;clear
myDir.dataviz = '/Users/wenwen/Documents/MatlabToolbox/Data-Viz';
addpath(genpath(myDir.dataviz))

MNIcoord = [18 9 18]; % right STA, note that andy updated his script and left-right was flipped
MNIcoord_control = [-6 9 -9]; % left STA

md_dir = '/Users/wenwen/Downloads/STA/CurrentFlowSimulation';
%% read csv

csv_files = dir(fullfile(md_dir,'*mm.csv'));
md_tbl = table;
for r = 1:length(csv_files)
    tmp_tbl = readtable(fullfile(md_dir,csv_files(r).name));
    if contains(csv_files(r).name,'con')
        tmp_tbl.mni = repmat(MNIcoord_control,height(tmp_tbl),1);
        tmp_tbl.center = repmat(2,height(tmp_tbl),1);
    else
        tmp_tbl.mni = repmat(MNIcoord,height(tmp_tbl),1);
        tmp_tbl.center = ones(height(tmp_tbl),1);
    end
    md_tbl = [md_tbl;tmp_tbl];
end
%%

myROI.r = sort(unique(md_tbl.radius_mm));
myROI.rn = length(myROI.r);

subs = table;
subs.name = sort(unique(md_tbl.subject));

%% read demographics
rest_tsv = fullfile(md_dir,'participants.tsv');
MRIsum = tdfread(rest_tsv);
MRIsum = struct2table(MRIsum);
MRIsum.participant_id = extractAfter(string(MRIsum.participant_id), 4);
MRIsum = sortrows(MRIsum,'participant_id');
[a,tmp_idx] = ismember(md_tbl.subject,cellstr(MRIsum.participant_id));
md_tbl.age = MRIsum.age(tmp_idx);
md_tbl.gender = MRIsum.gender_text(tmp_idx);
md_tbl.gender_code = MRIsum.gender_code(tmp_idx);

tmp_idx = matches(cellstr(MRIsum.participant_id),subs.name);
subs.age = MRIsum.age(tmp_idx);
subs.gender = MRIsum.gender_text(tmp_idx);
subs.gender_code = MRIsum.gender_code(tmp_idx);

%% excluding subjects
rows_idx = ismember(md_tbl.subject,{'CC320621','CC310252','CC321595','CC320321','CC312058','CC310252','CC420464','CC420392','CC520239','CC520215','CC520168','CC510480','CC621184','CC721532','CC711245'});
md_tbl(rows_idx,:) = [];
rows_idx = ismember(subs.name,{'CC320621','CC310252','CC321595','CC320321','CC312058','CC310252','CC420464','CC420392','CC520239','CC520215','CC520168','CC510480','CC621184','CC721532','CC711245'});
subs(rows_idx,:) = [];

%% excluding participants whose mean and std are 0

rows_idx = md_tbl.mean == 0 | md_tbl.std ==0;
zeo_subjects = unique(md_tbl.subject(rows_idx));
sortrows(md_tbl(ismember(md_tbl.subject,zeo_subjects),:),'subject')

rows_idx = ismember(md_tbl.subject,zeo_subjects);
md_tbl(rows_idx,:) = [];
disp(zeo_subjects)

rows_idx = ismember(subs.name,zeo_subjects);
subs(rows_idx,:) = [];

subN = height(subs);

mag_factor = 1.2; % right STA = 1.2, left STA = 1.6

md_tbl.mean = md_tbl.mean*mag_factor;
%% plot intensity and asymmetry of all

myStr.loc = {sprintf('Target\\newline%d %d %d', MNIcoord);sprintf('Control\\newline%d %d %d', MNIcoord_control)};
myStr.radius = arrayfun(@(x) sprintf('%d mm', x), myROI.r, 'UniformOutput', false);

grp_tbl = grpstats(md_tbl,["center","radius_mm"],["mean","sem"],"DataVars","mean");
figure('Position',[10 10 250*myROI.rn 600]);
for m = 1:myROI.rn
    subplot(2,myROI.rn,m);hold on;
    tmp_tbl = md_tbl(md_tbl.radius_mm==myROI.r(m),:);
    data2plot = table2array(tmp_tbl(:,["mean","center"]));

    subs.asym(:,m) = tmp_tbl.mean(tmp_tbl.center==1)./tmp_tbl.mean(tmp_tbl.center==2);
    h = daviolinplot(data2plot(:,1),'groups',data2plot(:,2),...
        'violin','full',...
        'boxcolors','w');
    avg2plot = grp_tbl.mean_mean(grp_tbl.radius_mm==myROI.r(m));
    plot(1:2,avg2plot,'k')

    [~,pval,~,stat] = ttest(tmp_tbl.mean(tmp_tbl.center==1),tmp_tbl.mean(tmp_tbl.center==2));
    plot(1:2,[1 1]*0.2,'k','HandleVisibility','off')

    text(1.5,0.2,'***','HorizontalAlignment','center','VerticalAlignment','bottom');
    cohend = diff(avg2plot)/stat.sd;
    text(1,0.9,sprintf('%.3f vs %.3f V/m\nt(%d)=%.3f\np=%.3f\nd=%.3f',avg2plot(1),avg2plot(2),stat.df,stat.tstat,pval,cohend),'sc','HorizontalAlignment','right')
    set(gca,'xtick',[1 2],'xlim',[0.5 2.5],'xticklabels',myStr.loc);
    ylabel('Modulation depth (V/m)');
    set(gca,'YTick',0:0.05:0.5,'YLim',[0 0.25])
    title(myStr.radius{m})
end

clear data2plot

subplot(2,3,4);
data2plot(:,1) = md_tbl.mean(md_tbl.center==1);
data2plot(:,2) = md_tbl.radius_mm(md_tbl.center==1);
h = daviolinplot(data2plot(:,1),'groups',data2plot(:,2),...
    'violin','full',...
    'boxcolors','w');
avg2plot = grp_tbl.mean_mean(grp_tbl.center==1);
plot(1:myROI.rn,avg2plot,'k')
set(gca,'xticklabel',myStr.radius)
ylabel('Target ROI intensity');
title(sprintf('N=%d',subN));

subplot(2,3,5);
data2plot(:,1) = md_tbl.mean(md_tbl.center==2);
data2plot(:,2) = md_tbl.radius_mm(md_tbl.center==2);
h = daviolinplot(data2plot(:,1),'groups',data2plot(:,2),...
    'violin','full',...
    'boxcolors','w');
avg2plot = grp_tbl.mean_mean(grp_tbl.center==2);
plot(1:myROI.rn,avg2plot,'k')
set(gca,'xticklabel',myStr.radius)
ylabel('Control ROI intensity');
title(sprintf('N=%d',subN));

subplot(2,3,6);
data2plot(:,1) = reshape(subs.asym,[],1);
data2plot(:,2) = sort(repmat([1:myROI.rn]',subN,1));
h = daviolinplot(data2plot(:,1),'groups',data2plot(:,2),...
    'violin','full',...
    'boxcolors','w');
plot(1:myROI.rn,mean(subs.asym),'k')
set(gca,'xticklabel',myStr.radius)
ylabel('Asymmetry ratio');
title(sprintf('N=%d',subN));

%% half-violin plot by age bins

ticks = round(prctile(subs.age,0:20:100));
clear myAgeGroups
edges = ticks;
for i = 1:length(edges)-1
    myAgeGroups{i} = [edges(i), edges(i+1)];
end
myAgeGroups{end+1} = [edges(1) edges(end)];

age_groupN = numel(myAgeGroups);
colors = lines(age_groupN);

md_tbl.age_group= discretize(md_tbl.age, edges);
subs.age_group= discretize(subs.age, edges);

control_region_file = fullfile(md_dir,'roi_means.csv');
control_regions = readtable(control_region_file);
tmp = startsWith(control_regions.subject,subs.name);
control_regions = control_regions(tmp,:);
control_regions{:, vartype('numeric')} = control_regions{:, vartype('numeric')} .* mag_factor;

locStr = {'Target','Right_Hippocampus','Right_Thalamus','Right_Amygdala',...
    'Cor7_Right','Cor3_Left'};

subs = [subs control_regions(:,locStr(2:end))];

legendLabels = cell(age_groupN,1);
for g = 1:age_groupN
    legendLabels{g} = sprintf('%d-%dyrs', myAgeGroups{g}(1), myAgeGroups{g}(2));
end
grp_tbl_age = grpstats(md_tbl,["center","radius_mm","age_group"],["mean","sem"],"DataVars","mean");


tbl2plot = md_tbl(md_tbl.center==1 & md_tbl.radius_mm==10,:);

decade_edges = [17 30:10:90];    % bin edges
labels = strcat(string(decade_edges(1:end-1)), '-', string(decade_edges(2:end)-1));
labels = cellstr(labels);
tbl2plot.ageBin = discretize(tbl2plot.age, decade_edges);
% Discretize ages into bins
tbl2plot.age_group = labels(discretize(tbl2plot.age, decade_edges))';

repSubj = {'CC110056'};
%
tbl2plot.age_group = categorical(tbl2plot.age_group, labels, 'Ordinal', true);
myColors = brewermap(numel(decade_edges)+2,'Blues');
myColors(1,:) = []; % avoid white

figure('Position',[10 10 450 200]);hold on; box off;
ug = categories(tbl2plot.age_group);
K  = numel(ug);
x = normalize(tbl2plot.age,'range',[1-0.4 K+0.4]);
y = tbl2plot.mean;
scatter(x, y, 5,'o', 'MarkerFaceColor', [0.85 0.85 0.88],'markeredgecolor',[0.85 0.85 0.88],'MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.05);

daviolinplot(tbl2plot.mean, ...
    'groups', tbl2plot.age_group, ...
    'violinwidth',2,'violin', 'half', 'violinalpha', 0.8, ...
    'scattercolors','same', 'scatter', 0,'scattersize',8, ...
    'xtlabels',ug,'withinlines',1,'linkline',1,...
    'colors',myColors,'jitter',1,...
    'boxcolors','same','boxalpha',0.8,'outliers', 0); 

row_idx = ismember(subs.name, repSubj);
scatter(1, tbl2plot.mean(row_idx), 30, 'b', 'o');

set(gca,'YLim',[0 0.5],'XTickLabel',ug,'XLim',[0.5 K+0.5])
ylabel('Mod. amp. (V/m)');
