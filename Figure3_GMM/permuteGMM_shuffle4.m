function gm_out = permuteGMM_shuffle4(gtbl, myFit)
% Function to fit Gaussian Mixture Model (GMM) for each group and phase
% Outputs: gm_out
% gtbl: Table with 'group', 'RTnorm', and 'phase' columns
% myFit: Structure containing fitting parameters (binN4group_perpart, norm, etc.)

% between groups: compare stimulation group versus sham, shuffle group labels:
% 1. pre; 2. stim-; 3. post-

% within -groups: between -phase
% 1: pre vs stim; 2, post vs stim;3: pre vs post;

unique_groups = unique(gtbl.group); % Get unique groups
com_pN = 3;
gm_out = struct;

[peak_shuf,weight_shuf] = deal(nan(length(unique_groups),com_pN,myFit.gaussN));% how many comparsions
auc_shuf = nan(length(unique_groups),com_pN,3);
compPhase = {[1 2],[2 3],[1 3]};
for gi = 2:length(unique_groups)

    for com_p = 1:com_pN

        rand_tblA1 = gtbl(ismember(gtbl.group,unique_groups([1 gi])) & ismember(gtbl.phase,compPhase{com_p}), :);
        a1 = sum(gtbl.group==unique_groups(1) & ismember(gtbl.phase,compPhase{com_p}(1)));
        a2 = sum(gtbl.group==unique_groups(1) & ismember(gtbl.phase,compPhase{com_p}(2)));
        b1 = sum(gtbl.group==unique_groups(gi) & ismember(gtbl.phase,compPhase{com_p}(1)));
        b2 = sum(gtbl.group==unique_groups(gi) & ismember(gtbl.phase,compPhase{com_p}(2)));
        arr = [repelem(1, a1), repelem(2, a2), repelem(3, b1), repelem(4, b2)];
        s_arr = arr(randperm(length(arr)));

        for i = 1:4
            y_data = rand_tblA1.RTnorm(s_arr==i);
            gms{i} = processGMM(y_data,myFit.gaussN, myFit);
        end
        %%

        weight_shuf(gi,com_p,:) = (gms{4}.weights-gms{3}.weights) - (gms{2}.weights-gms{1}.weights);
        peak_shuf(gi,com_p,:) = (gms{4}.peaks-gms{3}.peaks) - (gms{2}.peaks-gms{1}.peaks);
        auc_shuf(gi,com_p,:,:) = (gms{4}.auc-gms{3}.auc) - (gms{2}.auc-gms{1}.auc);
        sigma_shuf(gi,com_p,:,:) = (gms{4}.sigma-gms{3}.sigma) - (gms{2}.sigma-gms{1}.sigma);
        auc_1sd_component_shuf(gi,com_p,:,:) = (gms{4}.auc_1sd_component-gms{3}.auc_1sd_component) - (gms{2}.auc_1sd_component-gms{1}.auc_1sd_component);

    end
end
gm_out.weight = weight_shuf;
gm_out.peak = peak_shuf;
gm_out.auc = auc_shuf;
gm_out.auc_1sd_component_shuf = auc_1sd_component_shuf;
gm_out.sigma = sigma_shuf;
