clear;
load('/Users/wenwen/Downloads/STA/Behav/Exp1_RT.mat')

myMarkers = {'d','o','s','^'};
myColors = [ones(1,3)*0.5;lines(3)];% making sham = gray
group_id = unique(subs.group);
groupN = length(group_id);
phaseLabel = {'Pre','Stim','Post'};
groupLabel = {'Sham','RightBeta','LeftBeta'};
trl = [1, 2, 3, 4, 5, 6];
    exp_model = @(b, t) b(1) * exp(-b(2) * t) + b(3);

% reduce overlapping
jitter = (rand(1,groupN)-0.5)*0.5;
clear group_params
figure('Position',[100 10 900 150]);

for g = 1:groupN

    tmp_subs = subs.group==group_id(g) & ~out_idx; % excluding bad fitting

    for p = 1:3

        rt_valid = squeeze(mean(RTsubs(tmp_subs,p,:,:),[1 3],'omitnan'));
        se = squeeze(std(mean(RTsubs(tmp_subs,p,:,:),3,'omitnan'),0,1))./sqrt(sum(tmp_subs));

        initial_guess = [max(rt_valid), 0.1, min(rt_valid)];  % Adjust based on data
        lb = [0, 0, -Inf];  % Lower bounds for a, b, and c (b >= 0)
        ub = [Inf, Inf, max(rt_valid)];  % Upper bounds for a, b, and c (a, b, c positive)

        [fit_params,~,~,~,output] = nlinfit(trl, rt_valid', exp_model, initial_guess);
        group_params(g,p,:) = fit_params;

        subplot(1,3,p);hold all;

        plot(trl+jitter(g), rt_valid,'k','Marker',myMarkers{g}, 'MarkerSize', 6,'MarkerFaceColor',myColors(g,:),'LineStyle','none');
        errorbar(trl+jitter(g), rt_valid,se,'k', 'LineStyle','none','CapSize',0,'HandleVisibility','off');

        fitted_rt = exp_model(group_params(g,p,:), trl);
        plot(trl+jitter(g), fitted_rt, '-', 'LineWidth', 1, 'Color', myColors(g,:),'HandleVisibility','off');

        xlabel('Repetition');
        ylabel('Reaction time (s)');
        legend(groupLabel,'Location','bestoutside');
        title(phaseLabel{p})
        set(gca,'XLim',[0 7],'XTick',1:6,'ytick',[1:0.2:2])
        if p==1
            set(gca,'YLim',[1.1 1.8])
        else
            set(gca,'YLim',[1 1.6])
        end
    end
end
