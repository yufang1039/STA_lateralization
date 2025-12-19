clear
load('/Users/wenwen/Downloads/STA/Behav/STAResults.mat');

%% prepare data

grp4Exp1 = [1 5 7];
myStr.cond_exp1 = {'Sham','RightBeta','LeftBeta'};
groupN = length(grp4Exp1);
grp_subs = ismember(subs.group,grp4Exp1);
subs = subs(grp_subs,:);
gtbl = gtbl(ismember(gtbl.group,grp4Exp1),:);
subN = height(subs);

myMarkers.cond = {'d','o','s','^','o','^'};
myColors.block  = crameri('-oslo',9);
myColors.block = myColors.block(2:end,:);
myColors.repi = brewermap(7,'GnBu');
myColors.repi = myColors.repi(2:end,:); % skip the first color, almost transparent
myColors.cond = [ones(1,3)*0.5;brewermap(max(grp4Exp1),'set2')];% making sham = gray

myStr.tasks = {'Face';'Vehicle';'Motion'};
myStr.block = {'Block1','Block2','Block3','Block4','Block5','Block6'};
myStr.phase = {'Pre','Stim','Post'};
myStr.repi = {'Repi1','Repi2','Repi3','Repi4','Repi5','Repi6'};
myStr.cluster = {'Cluster1','Cluster2'};

% normalize RT, at the block level
for sub_i = 1:height(subs)
    for p = 1:3
        for b = 1:6
            tmp_idx = matches(gtbl.name,subs.name(sub_i)) & gtbl.phase == p & gtbl.block == b;
            gtbl.RTnorm(tmp_idx) = normalize(gtbl.RT(tmp_idx),'zscore');
        end
    end
end

myFit.norm = {'count','pdf'};
myFit.binN4group = 600;
myFit.binN4group_perpart = 300;
myFit.gaussNumArray = [2 3 4];
myFit.modelN = length(myFit.gaussNumArray);
myFit.gaussN = 2;
myFit.gaussType = ['gauss' num2str(myFit.gaussN)];
myFit.options =  statset('MaxIter',10000);

%% baseline across groups
rng(1)
y = gtbl.RTnorm(gtbl.phase==1);

figure('Position',[100 100 180 200]);   hold all;

GMModels = fitgmdist(y,myFit.gaussN,'Options',myFit.options);

[N, edges,bin] = histcounts(y,myFit.binN4group,'Normalization',myFit.norm{2});
x = (edges(1:end-1)+edges(2:end))/2;
plot(x,N,'k.','HandleVisibility','off','Color',[0.5 0.5 0.5]);

total_pdf = pdf(GMModels, y);
plot(y, total_pdf, 'k.','MarkerSize', 1,'Color',[0.5 0.5 0.5]); % Plot total GMMtmp_pdf

legendLabels = {'Data Histogram','Total GMM'};
[mu ,I]= sort(GMModels.mu);
sigma = GMModels.Sigma(I); % Covariances of the components
weights = GMModels.ComponentProportion(I); % Weights of the components

clear tmp_pdf
for i = 1:myFit.gaussN
    tmp_pdf{i} = weights(i) * normpdf(y, mu(i), sqrt(sigma(i))); % PDF of component i
    plot(y, tmp_pdf{i}, '.', 'MarkerSize', 1,'Color',myColors.block(i*3,:)); % Plot component
    legendLabels{end+1} = sprintf('Component %d', i); % Add label for the component
end
xlabel('RT (z)');
ylabel('Probability density function');
set(gca,'YTick',0:0.2:1)

%% baseline trials in each gaussian cluster;
% to reduce confounding from individual speed difference, below is
% based on RTnorm, which is zscore of each block
rng(2)
bs_tbl = gtbl(gtbl.phase==1,:);
y = bs_tbl.RTnorm;

gm = fitgmdist(y,myFit.gaussN,'Options',myFit.options);
myFit.Start = struct('mu',mu,'Sigma',gm.Sigma(I),'ComponentProportion',gm.ComponentProportion(I));
gms = processGMM(y, myFit.gaussN, myFit);
auc = gms.auc;
mu = gms.mu;
sigma = gms.sigma; % Covariances of the components
weights = gms.weights; % Weights of the components
bs_tbl.cluster = gms.cluster;

figure('Position',[100 100 1200 800]);
subplot(1,3,1);hold all;axis square

total_pdf = pdf(gm, y);

n = 2;% myFit.norm = {'count','pdf'}
[N, edges,bin] = histcounts(y,myFit.binN4group,'Normalization',myFit.norm{n});
x = (edges(1:end-1)+edges(2:end))/2;
plot(x,N,'b.','HandleVisibility','off');
plot(y, total_pdf, 'k.');
legendLabels = {'Data points'};

clear tmp_pdf
for i = 1:myFit.gaussN
    tmp_pdf{i} = weights(i) * normpdf(y, mu(i), sqrt(sigma(i))); % PDF of component i
    plot(y, tmp_pdf{i}, '.', 'LineWidth', 2); % Plot component
    legendLabels{end+1} = sprintf('Component %d', i); % Add label for the component
end
xlabel('RT (zscores)');
ylabel('Probability Density Function');

title('Gaussian Mixture Model Components',sprintf('N=%d',myFit.gaussN));
legend(legendLabels);

catVar = {'group','block','repi'};

for d = 1:2

    if d ==3
        plot_tbl = bs_tbl;
        tmp_title = 'all';
    else
        plot_tbl = bs_tbl(bs_tbl.cluster==d,:);
        bs_tbl.(['cluster',num2str(d)]) = cluster(gm,y)==d;
        tmp_title = ['Cluster ', num2str(d),': ',num2str(height(plot_tbl)),'trials'];
    end

    for c = 1:size(catVar,2)
        subplot(length(catVar),3,1+d+(c-1)*3);hold all;box on;
        if strcmp(catVar{c},'block')
            legendLabels = myStr.block ;
            tmp_color = myColors.block;
        elseif strcmp(catVar{c},'repi')
            legendLabels = myStr.repi;
            tmp_color = myColors.repi;
        else
            legendLabels = myStr.cond_exp1;
            tmp_color = myColors.cond;
        end

        if strcmp(catVar{c},'group')
            [chiRes.stat, chiRes.p, tmp_ratio] = deal(nan(groupN,1));
            for b = 1:groupN
                N = height(plot_tbl);  % Total number of observations (example)
                tmp_idx = plot_tbl.(catVar{c})==grp4Exp1(b);
                plot(linspace(1,100,sum(tmp_idx)),plot_tbl.RT(tmp_idx),'.','Color',tmp_color(b,:))
                tmp_ratio(b) = sum(tmp_idx)/N;

                % chi2 test
                n1 = sum(tmp_idx); N1 = N;
                n2 = round(1/numel(grp4Exp1)*N); N2 = N;
                x1 = [repmat('a',N1,1); repmat('b',N2,1)];
                x2 = [ones(n1,1); repmat(2,N1-n1,1); ones(n2,1); repmat(2,N2-n2,1)];
                [~,chiRes.stat(b),chiRes.p(b)] = crosstab(x1,x2);
            end
        elseif strcmp(catVar{c},'curTask')
            [chiRes.stat, chiRes.p, tmp_ratio] = deal(nan(3,1));
            for b = 1:3
                N = height(plot_tbl);  % Total number of observations (example)
                tmp_idx = plot_tbl.(catVar{c})==b;
                plot(linspace(1,100,sum(tmp_idx)),plot_tbl.RT(tmp_idx),'.','Color',tmp_color(b,:))
                tmp_ratio(b) = sum(tmp_idx)/N;

                % chi2 test
                n1 = sum(tmp_idx); N1 = N;
                n2 = round(1/3*N); N2 = N;
                x1 = [repmat('a',N1,1); repmat('b',N2,1)];
                x2 = [ones(n1,1); repmat(2,N1-n1,1); ones(n2,1); repmat(2,N2-n2,1)];
                [~,chiRes.stat(b),chiRes.p(b)] = crosstab(x1,x2);
            end

        else
            [chiRes.stat, chiRes.p, tmp_ratio] = deal(nan(6,1));
            for b = 1:6
                N = height(plot_tbl);  % Total number of observations (example)
                tmp_idx = plot_tbl.(catVar{c})==b;
                plot(linspace(1,100,sum(tmp_idx)),plot_tbl.RT(tmp_idx),'.','Color',tmp_color(b,:))
                tmp_ratio(b) = sum(tmp_idx)/N;

                % chi2 test
                n1 = sum(tmp_idx); N1 = N;
                n2 = round(1/6*N); N2 = N;
                x1 = [repmat('a',N1,1); repmat('b',N2,1)];
                x2 = [ones(n1,1); repmat(2,N1-n1,1); ones(n2,1); repmat(2,N2-n2,1)];
                [~,chiRes.stat(b),chiRes.p(b)] = crosstab(x1,x2);
            end
        end

        ratioStr =  arrayfun(@(i) sprintf('%s=%.3f, p=%.3f', legendLabels{i}, tmp_ratio(i), chiRes.p(i)),1:length(legendLabels),'UniformOutput',false);
        ratioStr = strjoin(ratioStr, '\n');
        text(1.02,0.2,ratioStr,'sc','fontsize',8);

        ylabel('RT (s)');xlabel('Data points');
        set(gca,'xtick',[],'XLim',[-1 101]);
        title(catVar{c},tmp_title);
        legend(legendLabels,'Location','bestoutside');
        legend('boxoff')
    end
end

%% scatterhist plot of cluster 1 and cluster 2

for d = 1:2 % each cluster

    plot_tbl = bs_tbl(bs_tbl.cluster==d,:);
    tmp_title = ['Cluster ', num2str(d)];

    for c = 3%1:size(catVar,2)
        figure('Position',[10 10 500 300]);
        if strcmp(catVar{c},'block')
            legendLabels = myStr.block ;
            tmp_color = myColors.block;
        elseif strcmp(catVar{c},'repi')
            legendLabels = myStr.repi;
            tmp_color = myColors.repi;
        else
            legendLabels = myStr.cond_exp1;
            tmp_color = myColors.cond;
        end
        [chiRes.stat, chiRes.p, tmp_ratio] = deal(nan(6,1));
        for b = 1:6
            N = height(plot_tbl);  % Total number of observations (example)
            tmp_idx = plot_tbl.(catVar{c})==b;
            tmp_ratio(b) = sum(tmp_idx)/N;

            % chi2 test
            n1 = sum(tmp_idx); N1 = N;
            n2 = round(1/6*N); N2 = N;
            x1 = [repmat('a',N1,1); repmat('b',N2,1)];
            x2 = [ones(n1,1); repmat(2,N1-n1,1); ones(n2,1); repmat(2,N2-n2,1)];
            [ct_tbl,chiRes.stat(b),chiRes.p(b)] = crosstab(x1,x2);

        end
        h = scatterhist(linspace(1,100,height(plot_tbl)),plot_tbl.RT,'Group',plot_tbl.(catVar{c}),'Kernel','on','Marker','.','NBins',myFit.binN4group_perpart*[1 1],'Color',tmp_color);
        h(3).Position = h(3).Position + [0.7 0 -0.1 0];
        delete(h(2))

        ratioStr =  arrayfun(@(i) sprintf('%s=%.3f, p=%.3f', legendLabels{i}, tmp_ratio(i), chiRes.p(i)),1:length(legendLabels),'UniformOutput',false);
        ratioStr = strjoin(ratioStr, '\n');
        text(1.2,-0.3,ratioStr,'sc','fontsize',8);

        ylabel('RT (s)');xlabel('Data points');
        set(gca,'xtick',[]);
        title(catVar{c},tmp_title);
        legend(legendLabels,'Location','bestoutside','Box','off');
    end
end

%% monotonic change with repetitions
figure('Position',[10 10 600 200]);

for c = 1:size(catVar,2)
    if strcmp(catVar{c},'block')
        legendLabels = myStr.block ;
        tmp_color = myColors.block;
    elseif strcmp(catVar{c},'repi')
        legendLabels = myStr.repi;
        tmp_color = myColors.repi;
    else
        legendLabels = myStr.cond_exp1;
        tmp_color = myColors.cond;
    end

    for d = 1:2 % each cluster

        plot_tbl = bs_tbl(bs_tbl.cluster==d,:);
        tmp_title = ['Cluster ', num2str(d)];

        [chiRes.stat, chiRes.p, tmp_ratio] = deal(nan(6,1));
        clear tmp_ratio
        for b = 1:length(legendLabels)
            N = height(plot_tbl);  % Total number of observations (example)
            if strcmp(catVar{c},'group')
                tmp_idx = plot_tbl.(catVar{c})==grp4Exp1(b);
            else
                tmp_idx = plot_tbl.(catVar{c})==b;
            end
            tmp_ratio(b) = sum(tmp_idx)/N;

            % chi2 test
            n1 = sum(tmp_idx); N1 = N;
            n2 = round(1/length(legendLabels)*N); N2 = N;
            x1 = [repmat('a',N1,1); repmat('b',N2,1)];
            x2 = [ones(n1,1); repmat(2,N1-n1,1); ones(n2,1); repmat(2,N2-n2,1)];
            [~,chiRes.stat(b),chiRes.p(b)] = crosstab(x1,x2);
        end

        subplot(2,size(catVar,2),c+(d-1)*size(catVar,2));hold all;
        h = bar(1,tmp_ratio);
        for b = 1:length(h)
            h(b).FaceColor = tmp_color(b,:);
            if chiRes.p(b)<0.05
                text(h(b).XEndPoints,tmp_ratio(b),'*',HorizontalAlignment='center')
            end
        end

        plot(get(gca,'XLim'),[1 1]/length(legendLabels),'k--','HandleVisibility','off')
        ratioStr =  arrayfun(@(i) sprintf('%s=%.3f, p=%.3f', legendLabels{i}, tmp_ratio(i), chiRes.p(i)),1:length(legendLabels),'UniformOutput',false);

        ylabel('Proportion');
        hl = legend(legendLabels,'Location','bestoutside','Box','off');
        set(gca,'xtick',[],'YTick',0.1:0.1:0.9,'ylim',[0.1 0.6]);
        if strcmp(catVar{c},'block')
            set(gca,'YLim',[0.1 0.2])
        elseif strcmp(catVar{c},'repi')
            set(gca,'YLim',[0.1 0.24],'YTick',[0.1 0.24])
        end
        title(tmp_title);
        xlabel(catVar{c})
    end
end
%% 2D plot of clustering results, repi x block

data2show = grpstats(bs_tbl,["block","repi"],["mean"],'datavar',"cluster1");
figure('Position',[10 10 200 160]);
% Create the heatmap with block (rows), repi (columns), and mean_cluster as values
h = heatmap(data2show,'repi','block','ColorVariable','mean_cluster1','Colormap',brewermap(20,'GnBu'));
h.YDisplayData = string(flip(1:max(data2show.block)));
h.XDisplayData = string(1:max(data2show.repi));
xlabel('Repetition');
ylabel('Block');
title('Mean Cluster1 Heatmap');
h.ColorLimits = [0.4 0.8];
h.ColorbarVisible = 'on';
h.CellLabelColor = 'none';

%% permutation to get parameter distribution

iter_n = 5000;
[gm_out, gm_sampled,gm_out_mixed,gm_out_shuffled] = deal(cell(iter_n,1));
parfor iter = 1:iter_n
    gm_out{iter} = permuteGMM(gtbl, myFit);
    gm_out_mixed{iter} = permuteGMM_mixed(gtbl, myFit); % this way of permutation would provide mean difference at similar scale of the bar plot
    gm_sampled{iter} = BoostrapGMM(gtbl,myFit);
    gm_out_shuffled{iter} = permuteGMM_shuffle4(gtbl, myFit)
    gm_out_slope{iter} = permuteGMM_slope(gtbl, myFit)
end

tmp_peak = cellfun(@(x) x.peak, gm_out_shuffled, 'UniformOutput', false);
peak_shuff = cat(4, tmp_peak{:});% group*phase*cluster2*iter

tmp = cellfun(@(x) x.weight, gm_out_shuffled, 'UniformOutput', false);
weight_shuff = cat(4, tmp{:});% group*phase*2*iter

tmp = cellfun(@(x) x.sigma, gm_out_shuffled, 'UniformOutput', false);
sigma_shuff = cat(4, tmp{:});% group*phase*2*iter

tmp_peak = cellfun(@(x) x.peak, gm_out_mixed, 'UniformOutput', false);
peak_mixed = cat(4, tmp_peak{:});% group*phase*cluster2*iter

tmp = cellfun(@(x) x.weight, gm_out_mixed, 'UniformOutput', false);
weight_mixed = cat(4, tmp{:});% group*phase*cluster2*iter

tmp = cellfun(@(x) x.sigma, gm_out_mixed, 'UniformOutput', false);
sigma_mixed = cat(4, tmp{:});% group*phase*cluster2*iter

tmp = cellfun(@(x) x.weight, gm_out_slope, 'UniformOutput', false);
perm_slope.weight = cat(3, tmp{:});% group*3*iter

tmp = cellfun(@(x) x.peak, gm_out_slope, 'UniformOutput', false);
perm_slope.peak = cat(3, tmp{:});% group*3*iter

tmp = cellfun(@(x) x.trial_prop_slope, gm_out_slope, 'UniformOutput', false);
perm_slope.trial_prop = cat(3, tmp{:});% group*3*iter


tmp = cellfun(@(x) x.peak, gm_sampled, 'UniformOutput', false);
peak_groups_sampled = cat(4, tmp{:});% group*phase*cluster2*iter

tmp = cellfun(@(x) x.sigma, gm_sampled, 'UniformOutput', false);
sigma_groups_sampled = cat(4, tmp{:});% group*phase*cluster2*iter

%% for each stim group
GausFit.peak = nan(groupN,3,myFit.gaussN);% group*phase*cluster
for n = 2 % two clusters
    for gi = 1:groupN
        plot_tbl = gtbl(gtbl.group==grp4Exp1(gi),:);
        for p = 1:3 % each part

            y = plot_tbl.RTnorm(plot_tbl.phase==p);

            gm = fitgmdist(y,myFit.gaussN,'Options',myFit.options,'Start',myFit.Start);
            gms = processGMM(y, myFit.gaussN, myFit);
            plot_tbl.cluster(plot_tbl.phase==p) = gms.cluster;

            GausFit.auc(gi,p,:) = gms.auc;
            GausFit.peak(gi,p,:) = gms.peaks;
            GausFit.sigma(gi,p,:) = gms.sigma;
            GausFit.weight(gi,p,:) = gms.weights;
            GausFit.mu(gi,p,:) = gms.mu;
            GausFit.auc_2sd(gi,p,:) = gms.auc_2sd_mixture;
            GausFit.auc_1sd(gi,p,:) = gms.auc_1sd_mixture;
            GausFit.post{gi,p} = gms.post_p(:,1);
        end

    end
end

%% test the difference of difference- peak
% 1. permute between stim and pre of Beta condition and get Stim minus
% shuffled as distribution A, which is 0.5*(stim-pre) of beta condition
% 2. permute between stim and pre of Sham condition and get Stim minus
% shuffled as distribution B, which is 0.5*(stim-pre) of sham condition
% 3. t-test of A versus B

clear pval
compStr = {'Pre vs Stim','Post vs Stim','Post vs Pre'};
thresh_z = norminv(1-0.05/2);

for gi = 2:groupN
    figure('position',[100 100, 800 600]);
    for p = 1:3
        for i = 1:2
            subplot(3,2,i+(p-1)*2);hold all

            perm_data = peak_shuff(gi,p,i,:);
            histogram(perm_data,'Normalization','probability')

            A = squeeze(peak_mixed(gi,p,i,:));
            mn = mean(A);
            sd = std(A);
            CI = [mn-thresh_z*sd, mn+thresh_z*sd];

            plot([1 1]*(mn+thresh_z*sd),get(gca,'ylim'),'k--','HandleVisibility','off');% lower
            plot([1 1]*(mn-thresh_z*sd),get(gca,'ylim'),'k--');% upper
            title(compStr{p},['cluster#',num2str(i)])

            if p ==1
                cond_diff = (GausFit.peak(gi,2,i) - GausFit.peak(1,2,i)) - (GausFit.peak(gi,1,i) - GausFit.peak(1,1,i));
            elseif p ==2
                cond_diff = (GausFit.peak(gi,3,i) - GausFit.peak(1,3,i)) - (GausFit.peak(gi,2,i) - GausFit.peak(1,2,i));
            else
                cond_diff = (GausFit.peak(gi,3,i) - GausFit.peak(1,3,i)) - (GausFit.peak(gi,1,i) - GausFit.peak(1,1,i));
            end
            plot([1 1]*cond_diff,get(gca,'ylim'),'k-','LineWidth',2);% mean

            pval(gi,p,i) = mean(abs(perm_data)>abs(cond_diff));
            permutedH =pval < 0.05;

            if permutedH(gi,p,i) ==1
                text(cond_diff,0.02,'*','color','r','HorizontalAlignment','center','FontSize',30)
            end

            ylabel('pdf')
            xlabel('Value')
            text(0.01,0.9,sprintf('cond diff = %.3f, SD = %.3f, CI = [%.3f,%.3f]\np = %.3f, cohenD = %.3f',cond_diff,sd,CI(1),CI(2),pval(gi,p,i),mn/sd),'sc')
            legend({'Permutation','95% CI','True diff'})
        end
    end
    sgtitle(myStr.cond_exp1{gi})
end

%% bar plot of peak, online and offline stim effect relative to sham
tmp = GausFit.peak(:,2:end,:) - GausFit.peak(:,1,:); % stim effect
clear stimEffect
stimEffect.peak = tmp - tmp(1,:,:); % minus baseline

myColors.peaks = brewermap(11,'RdBu');
myColors.peaks = myColors.peaks([6  8 10],:);
figure('Position', [100, 100, 800, 180])

for i = 1:2
    subplot(1,2,i)
    hold all;
    dat2plot = GausFit.peak(:,:,i);  % GroupN by phase
    set2plot = squeeze(std(peak_groups_sampled(:,:,i,:), 0, 4));  % Standard error

    b = bar(dat2plot,'FaceAlpha',1);
    for j = 1:3
        b(j).FaceColor = myColors.peaks(j,:);  % Set color for each bar
    end

    xcord = vertcat(b.XEndPoints)';
    errorbar(xcord, dat2plot, set2plot, 'k', 'LineStyle', 'none')
    ylabel(['Height of cluster ' num2str(i)])

    legend(myStr.phase,'Location','northeastoutside','Box','off')
    set(gca, 'XTick', 1:groupN, 'xticklabel', myStr.cond_exp1,'xlim',[0 groupN+1])

    if i == 1
        set(gca, 'ylim', [0.2 0.8],'YTick',0:0.1:1)
    else
        set(gca, 'ylim', [0 0.4],'YTick',0:0.1:1)
    end

    % Inserted figure for stimulation effect
    axInset = axes('Position', [0.23+(i-1)*0.44, 0.75, 0.11, 0.19]); % Adjust position for inset
    dat2plot_inset = stimEffect.peak(:,:,i);  % Stim effect for the inset
    b_inset = bar(axInset, dat2plot_inset);
    xcord = vertcat(b_inset.XEndPoints);

    for gi = 2:groupN
        for p = [1 3]
            if permutedH(gi,p,i)==1
                text(xcord((p+1)/2,gi),0.1,'*','Color','r','FontSize',20,'HorizontalAlignment','center')
            end
        end
    end
    for j = 1:2
        b_inset(j).FaceColor = myColors.peaks(j+1,:);
    end
    set(axInset, 'XTick',[], 'box', 'on','xlim',[1 4])
    title(axInset, 'Stimulation effect')
end

%% bar plot of sigma, online and offline stim effect relative to sham
tmp= GausFit.sigma- GausFit.sigma(1,:,:);
stimEffect.sigma = tmp - tmp(:,1,:);

figure('Position', [100, 100, 800, 180])

for i = 1:2
    subplot(1,2,i)
    hold all;
    dat2plot = GausFit.sigma(:,:,i);  % GroupN by phase
    set2plot = squeeze(std(sigma_groups_sampled(:,:,i,:), 0, 4));  % Standard error

    b = bar(dat2plot,'FaceAlpha',1);
    for j = 1:3
        b(j).FaceColor = myColors.peaks(j,:);  % Set color for each bar
    end

    xcord = vertcat(b.XEndPoints)';
    errorbar(xcord, dat2plot, set2plot, 'k', 'LineStyle', 'none')
    ylabel(['Sigma of cluster ' num2str(i)])

    legend(myStr.phase,'Location','northeastoutside','Box','off')
    set(gca, 'XTick', 1:groupN, 'xticklabel', myStr.cond_exp1,'xlim',[0 groupN+1])
    if i == 1
        set(gca, 'ylim', [0 0.4],'YTick',0:0.1:1)
    else
        set(gca, 'ylim', [0.6 1.6],'YTick',0.6:0.2:2)
    end

    % Inserted figure for stimulation effect
    axInset = axes('Position', [0.23+(i-1)*0.44, 0.7, 0.11, 0.19]); % Adjust position for inset
    dat2plot_inset = stimEffect.sigma(:,:,i);  % Stim effect for the inset
    b_inset = bar(axInset, dat2plot_inset,'w');
    xcord = vertcat(b_inset.XEndPoints);

    % Color for inset bars
    for j = 2:3
        b_inset(j).FaceColor = myColors.peaks(j,:);
    end
    set(axInset, 'XTick',[], 'box', 'on','xlim',[1 4])
    title(axInset, 'Stimulation effect')
    if i == 1
        set(gca, 'ylim', [-0.05 0.1])
    else
        set(gca, 'ylim', [-0.1 0.31])
    end
end


%% get the slope of weight/auc changes across phases
myStr.varStr = {'weight','peak'};
for v = 1:length(myStr.varStr)
    for gi = 1:groupN
        tmp_dat = squeeze(GausFit.(myStr.varStr{v})(gi,:,:));

        for i = 1:size(tmp_dat,2)
            [a,S] = polyfit(1:3,tmp_dat(:,i),1);

            GausFit.([myStr.varStr{v},'_sl'])(gi,i,:) = a;
            GausFit.([myStr.varStr{v},'_S']){gi,i} = S;
        end
    end
    stimEffect.([myStr.varStr{v},'_sl']) = GausFit.([myStr.varStr{v},'_sl']) - GausFit.([myStr.varStr{v},'_sl'])(1,:,1);
end

%%  plot of weight, plot raw and linear fit

v = 1;
c = 1;
figure('position',[10 10 300 180]);
hold all;
for gi = 1:3

    plot(GausFit.(myStr.varStr{v})(gi,:,c),'marker',myMarkers.cond{gi},'markerEdgeColor', 'k','markerFacecolor',myColors.cond(gi,:),'color',myColors.cond(gi,:),'LineWidth', 1);
    x = 1:3;

    [y_fit, delta] = polyval(squeeze(GausFit.([myStr.varStr{v},'_sl'])(gi,c,:)), x,  GausFit.([myStr.varStr{v},'_S']){gi,c}); % y values and 95% CI

    y_upper = y_fit + delta;
    y_lower = y_fit - delta;

    % Plot the confidence interval as a shaded area
    fill([x, fliplr(x)], [y_upper, fliplr(y_lower)], ...
        [myColors.cond(gi,:)], 'EdgeColor', 'none','FaceAlpha', 0.3,'HandleVisibility','off'); % Light gray shading
    plot(x, y_fit, '--', 'LineWidth', 0.25,'color',myColors.cond(gi,:),'HandleVisibility','off');
end
legend(myStr.cond_exp1,'Location','eastoutside')
title(myStr.cluster{c})
ylabel(myStr.varStr{v})
set(gca,'xtick',1:3,'xticklabel',myStr.phase,'xlim',[0 4],'YLim',[0.45 0.8])

%% examine posterior of each trial being component 1 or 2

clear trial_prop_sl
figure('Position',[10 10 2000 1000]);

c1_thresh = 0.65:0.05:0.9;
for i = 1:length(c1_thresh)
    p_c1 = c1_thresh(i);

    trial_prop = cell2mat(cellfun(@(x) mean(x > p_c1), GausFit.post,'UniformOutput',false));% group * phase

    subplot(4,length(c1_thresh),i);
    axis square; hold on

    x = 1:3;
    for gi = 1:groupN
        [a,S] = polyfit(1:3,trial_prop(gi,:),1);
        trial_prop_sl(gi,:) = a(1);

        plot(trial_prop(gi,:),'--','marker',myMarkers.cond{gi},'markerEdgeColor', 'k','markerFacecolor',myColors.cond(gi,:),'color',myColors.cond(gi,:),'LineWidth', 1);

        [y_fit, delta] = polyval(a, x,  S); % y values and 95% CI

        residuals = trial_prop(gi,:) - y_fit;
        sd_res    = std(residuals);

        y_upper = y_fit + sd_res;
        y_lower = y_fit - sd_res;

        % Plot the confidence interval as a shaded area
        fill([x, fliplr(x)], [y_upper, fliplr(y_lower)], ...
            [myColors.cond(gi,:)], 'EdgeColor', 'none','FaceAlpha', 0.3,'HandleVisibility','off'); % Light gray shading
        plot(x, y_fit, '-', 'LineWidth', 2,'color',myColors.cond(gi,:),'HandleVisibility','off');
    end

    xlabel('Phase');
    ylabel('Proportion of component 1');
    title(sprintf('posterior > %.2f',p_c1))
    legend(myStr.cond_exp1,'Location','best','box','off')
    set(gca,'xtick',1:3,'xticklabel',myStr.phase,'xlim',[0 4],'ytick',0:0.1:1)

    subplot(4,length(c1_thresh),i+length(c1_thresh))

    diff_prop = trial_prop(2:end,:) - trial_prop(1,:);
    diff_prop = diff_prop(:,2:end) - diff_prop(:,1); % group * phase
    bar(diff_prop)
    set(gca,'ylim',[0 0.4])
    set(gca,'xticklabel',myStr.cond_exp1(2:end));
    legend(myStr.phase(2:end))

    for gi = 2:groupN
        perm_data = squeeze(perm_slope.trial_prop(gi,:,:));
        cn = size(perm_data,1);

        subplot(4,length(c1_thresh),i+length(c1_thresh)*gi);hold on
        histogram(perm_data(gi,:),'Normalization', 'pdf','HandleVisibility','off')

        cond_diff = trial_prop_sl(gi) - trial_prop_sl(1);
        plot([1 1]*cond_diff,get(gca,'YLim'),'b-')

        % pval = mean(abs(perm_data(gi,:))>abs(cond_diff));% two tailed
        pval = mean( perm_data(gi,:) > cond_diff ); % cond_diff > 0

        text(0,0.2,sprintf('%s, p = %.3f', myStr.cond_exp1{gi},pval),'sc')

        if pval < 0.05
            text(cond_diff,0.02,'*','color','r','HorizontalAlignment','center','FontSize',30)
        end
        title(myStr.cond_exp1(gi))

    end
end
%% remove baseline
figure('Position',[10 10 2000 250]);

c1_thresh = 0.65:0.05:0.9;
for i = 1:length(c1_thresh)
    p_c1 = c1_thresh(i);
    trial_prop = cell2mat(cellfun(@(x) mean(x > p_c1), GausFit.post,'UniformOutput',false));% group * phase
    trial_prop = trial_prop - trial_prop(:,1);
    subplot(1,length(c1_thresh),i);
    axis square; hold on

    x = 1:3;
    for gi = 1:groupN
        [a,S] = polyfit(1:3,trial_prop(gi,:),1);
        trial_prop_sl(gi,:) = a(1);

        plot(trial_prop(gi,:),'--','marker',myMarkers.cond{gi},'markerEdgeColor', 'k','markerFacecolor',myColors.cond(gi,:),'color',myColors.cond(gi,:),'LineWidth', 1);

        [y_fit, delta] = polyval(a, x,  S); % y values and 95% CI

        residuals = trial_prop(gi,:) - y_fit;
        sd_res    = std(residuals);

        y_upper = y_fit + sd_res;
        y_lower = y_fit - sd_res;

        % Plot the confidence interval as a shaded area
        fill([x, fliplr(x)], [y_upper, fliplr(y_lower)], ...
            [myColors.cond(gi,:)], 'EdgeColor', 'none','FaceAlpha', 0.3,'HandleVisibility','off'); % Light gray shading
        plot(x, y_fit, '-', 'LineWidth', 2,'color',myColors.cond(gi,:),'HandleVisibility','off');
    end

    xlabel('Phase');
    ylabel('Proportion of component 1');
    title(sprintf('posterior > %.2f',p_c1))
    legend(myStr.cond_exp1,'Location','best','box','off')
    set(gca,'xtick',1:3,'xticklabel',myStr.phase,'xlim',[0 4],'ytick',0:0.1:1)
end

