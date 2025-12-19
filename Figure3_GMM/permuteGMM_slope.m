function gm_out = permuteGMM_slope(gtbl, myFit)
% Function to fit Gaussian Mixture Model (GMM) for each group and phase
% Outputs: gm_out
% gtbl: Table with 'group', 'RTnorm', and 'phase' columns
% myFit: Structure containing fitting parameters (binN4group_perpart, norm, etc.)

unique_groups = unique(gtbl.group); % Get unique groups
nGroups = numel(unique_groups);

gm_out = struct;

[peak_slope,weight_slope,peak_slope_diff,weight_slope_diff] = deal(nan(nGroups,myFit.gaussN));% how many comparsions
[auc_slope,auc_slope_diff]= deal(nan(nGroups,3));

for p = 1:3
    shamN(p) = sum(gtbl.group == unique_groups(1) & gtbl.phase == p);
end

for gi = 2:nGroups
    % Extract sham and treatment group data
    tisGroup = unique_groups(gi);
    for p = 1:3
        tisN(p) = sum(gtbl.group == tisGroup & gtbl.phase == p);
    end

    selected_rows = ismember(gtbl.group, [unique_groups(1), tisGroup]);
    rand_tbl = gtbl(selected_rows, :);
    rand_tbl = rand_tbl(randperm(height(rand_tbl)),:);

     shamIdx = cumsum([0, shamN]);
    tisIdx  = cumsum([0, tisN])+sum(shamN);
    % Divide RTnorm into 3 parts each for sham and tis
    for i = 1:3
        a_data = rand_tbl.RTnorm(shamIdx(i)+1:shamIdx(i+1));
        b_data = rand_tbl.RTnorm(tisIdx(i)+1:tisIdx(i+1));

        gmsA{i} = processGMM(a_data, myFit.gaussN, myFit);
        gmsB{i} = processGMM(b_data, myFit.gaussN, myFit);

        tmp_weightA(i,:) = gmsA{i}.weights;
        tmp_weightB(i,:) = gmsB{i}.weights;

        tmp_peakA(i,:) = gmsA{i}.peaks;
        tmp_peakB(i,:) = gmsB{i}.peaks;

        tmp_aucA(i,:) = gmsA{i}.auc;
        tmp_aucB(i,:) = gmsB{i}.auc;

        tmp_auc_1sdA(i,:) = gmsA{i}.auc_1sd_component;
        tmp_auc_1sdB(i,:) = gmsB{i}.auc_1sd_component;

        c1_thresh = 0.65:0.05:0.9;
        for c = 1:length(c1_thresh)
            p_c1 = c1_thresh(c);

            tmp_propA(i,c) = mean(gmsA{i}.post_p(:,1)>p_c1);
            tmp_propB(i,c) = mean(gmsB{i}.post_p(:,1)>p_c1);
        end
    end

    % Fit linear
    x = 1:3;

    for c = 1:length(c1_thresh)
        a_fit = polyfit(x, tmp_propA(:,c), 1);
        b_fit = polyfit(x, tmp_propB(:,c), 1);

        trial_prop_slope(gi,c, :) = b_fit(1);
        trial_prop_slope_diff(gi,c, :) = b_fit(1) - a_fit(1);

    end
    for c = 1:2
        a_fit = polyfit(x, tmp_weightA(:,c), 1);
        b_fit = polyfit(x, tmp_weightB(:,c), 1);

        weight_slope(gi,c, :) = b_fit(1);
        weight_slope_diff(gi,c, :) = b_fit(1) - a_fit(1);

        a_fit = polyfit(x, tmp_peakA(:,c), 1);
        b_fit = polyfit(x, tmp_peakB(:,c), 1);
        peak_slope(gi,c, :) = b_fit(1);
        peak_slope_diff(gi,c, :) = b_fit(1) - a_fit(1);

        a_fit = polyfit(x, tmp_auc_1sdA(:,c), 1);
        b_fit = polyfit(x, tmp_auc_1sdB(:,c), 1);
        auc_1sd_slope(gi,c, :) = b_fit(1);
        auc_1sd_slope_diff(gi,c, :) = b_fit(1) - a_fit(1);
    end
    for c = 1:3
        a_fit = polyfit(x, tmp_aucA(:,c), 1);
        b_fit = polyfit(x, tmp_aucB(:,c), 1);
        auc_slope(gi,c, :) = b_fit(1);
        auc_slope_diff(gi,c, :) = b_fit(1) - a_fit(1);
    end
end

gm_out.weight = weight_slope;
gm_out.peak = peak_slope;
gm_out.auc = auc_slope;
gm_out.auc_1sd_slope = auc_1sd_slope;
gm_out.trial_prop_slope = trial_prop_slope;


gm_out.weight_diff = weight_slope_diff;
gm_out.peak_diff = peak_slope_diff;
gm_out.auc_diff = auc_slope_diff;
gm_out.auc_1sd_slope_diff = auc_1sd_slope_diff;
gm_out.trial_prop_slope_diff = trial_prop_slope_diff;
