function gm_out = permuteGMM_mixed(gtbl, myFit)
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

[peak_out,weight_out,sigma_out] = deal(nan(length(unique_groups),com_pN,myFit.gaussN));% how many comparsions

% Phase comparison pairs: [pre vs stim], [stim vs post], [pre vs post]
compPhase = {[1 2],[2 3],[1 3]};
for gi = 2:length(unique_groups)
    for com_p = 1:com_pN
        % Define comparison phases
        ph1 = compPhase{com_p}(1);
        ph2 = compPhase{com_p}(2);

        % Sample and ensure all participants are represented
        rand_tblA1 = sample_half_with_all_participants(gtbl, unique_groups(1), ph1);
        rand_tblA2 = sample_half_with_all_participants(gtbl, unique_groups(1), ph2);
        rand_tblB1 = sample_half_with_all_participants(gtbl, unique_groups(gi), ph1);
        rand_tblB2 = sample_half_with_all_participants(gtbl, unique_groups(gi), ph2);

        % Fit GMMs and get peaks
        data = {rand_tblA1.RTnorm, rand_tblA2.RTnorm, rand_tblB1.RTnorm, rand_tblB2.RTnorm};
        % [peaks,weight,sigma] = deal(zeros(1,4, myFit.gaussN));
               [peaks,weight,sigma] = deal(zeros(4, myFit.gaussN));
 for i = 1:4
            gms = processGMM(data{i}, myFit.gaussN, myFit);
            % peaks(1,i, :) = gms.peaks;% the size of the right side is 1-by-3.
            % weight(1,i,:) = gms.weights;
            % auc(1,i,:) = gms.auc;
            % sigma(1,i,:) = gms.sigma;

              gms = processGMM(data{i}, myFit.gaussN, myFit);
            peaks(i, :) = gms.peaks;% the size of the right side is 1-by-3.
            weight(i,:) = gms.weights;
            auc(i,:) = gms.auc;
            sigma(i,:) = gms.sigma;
        end

        % Compute peak difference
        peak_out(gi, com_p, :) = (peaks(4,:) - peaks(3,:)) - (peaks(2,:) - peaks(1,:));
        weight_out(gi, com_p, :) = (weight(4,:) - weight(3,:)) - (weight(2,:) - weight(1,:));
        auc_out(gi, com_p, :) = (auc(4,:) - auc(3,:)) - (auc(2,:) - auc(1,:));
        sigma_out(gi, com_p, :) = (sigma(4,:) - sigma(3,:)) - (sigma(2,:) - sigma(1,:));

        %  peak_out(gi, com_p, :) = (peaks(1,4,:) - peaks(1,3,:)) - (peaks(1,2,:) - peaks(1,1,:));
        % weight_out(gi, com_p, :) = (weight(1,4,:) - weight(1,3,:)) - (weight(1,2,:) - weight(1,1,:));
        % auc_out(gi, com_p, :) = (auc(1,4,:) - auc(1,3,:)) - (auc(1,2,:) - auc(1,1,:));
        % sigma_out(gi, com_p, :) = (sigma(1,4,:) - sigma(1,3,:)) - (sigma(1,2,:) - sigma(1,1,:));


    end
end

gm_out.peak = peak_out;
gm_out.weight = weight_out;
gm_out.auc = auc_out;
gm_out.sigma = sigma_out;

end

function sampled_tbl = sample_half_with_all_participants(gtbl, group_label, phase_label)
% Randomly sample 50% of trials while retaining all participants
tbl = gtbl(gtbl.group == group_label & gtbl.phase == phase_label, :);
unique_names = unique(tbl.name);
while true
    idx = randperm(height(tbl), round(height(tbl)*0.5));
    sampled = tbl(idx, :);
    if length(unique(sampled.name)) == length(unique_names)
        sampled_tbl = sampled;
        return
    end
end
end
