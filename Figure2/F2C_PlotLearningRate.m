
figure('Position',[20 20 300 240]);
hold all
chart_tbl = readtable('/Users/wenwen/Downloads/STA/Behav/Exp1_lrate.csv');
group_id = unique(chart_tbl.group);
phaseLabel = {'Pre','Stim','Post'};
groupLabel = {'Sham','RightBeta','LeftBeta'};
for gi = 1:3
    tmp_grp_tbl = chart_tbl(chart_tbl.group==group_id(gi),:);

    mdl = fitlm(tmp_grp_tbl, 'data ~ phase');

    % Predict with confidence intervals
    xfit = linspace(1, 3, 50)';
    [yfit, yci] = predict(mdl, table(xfit, 'VariableNames', {'phase'}));

    xplot = normalize(xfit,'range',tmp_grp_tbl.xvalue(1)+[-0.5 0.5]);
    fill([xplot; flipud(xplot)], ...
        [yci(:,1); flipud(yci(:,2))], ...
        [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none','HandleVisibility','off');
    plot(xplot, yfit, '-', 'LineWidth', 1, 'Color', [0.5 0.5 0.5],'HandleVisibility','off');
end

b=boxchart(chart_tbl.xvalue,chart_tbl.data,'GroupByColor',chart_tbl.phase,'BoxWidth',0.8,'MarkerStyle','none');

legend(phaseLabel,'Location','northeastoutside')
set(gca,'XTick',1:2:5,'XTickLabel',groupLabel)
set(gca,'YLim',[-2 6],'YTick',-6:2:20)
ylabel('learning rate')


