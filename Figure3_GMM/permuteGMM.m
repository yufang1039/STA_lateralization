function gm_out = permuteGMM(gtbl, myFit)
% Function to fit Gaussian Mixture Model (GMM) for each group and phase
% Outputs: gm_phase_out, mu_out, sigma_out, gm_out.w_group
% gtbl: Table with 'group', 'RTnorm', and 'phase' columns
% myFit: Structure containing fitting parameters (binN4group_perpart, norm, etc.)


% between groups: compare stimulation group versus sham, shuffle group labels:
% 1. pre; 2. stim-; 3. post-

% within -groups: between -phase
% 1: pre vs stim; 2, post vs stim;3: pre vs post;

unique_groups = unique(gtbl.group); % Get unique groups
com_pN = 3;
gm_out = struct;
[mu_out,sigma_out,weight_out,peak_out] = deal(nan(length(unique_groups),3,myFit.gaussN));% how many comparsions

for gi = 2:length(unique_groups) % the other group is always sham

    for p = 1:3
        % select half data from each group
        while 1
            rand_tbl1 = gtbl(gtbl.group==1 & gtbl.phase ==p, :);
            tmp_rand1 = randperm(height(rand_tbl1),round(height(rand_tbl1)*0.5)); % select half of the data
            rand_tbl2 = gtbl(gtbl.group==unique_groups(gi) & gtbl.phase ==p, :);
            tmp_rand2 = randperm(height(rand_tbl2),round(height(rand_tbl2)*0.5)); % select half of the data
            rand_tbl = [rand_tbl1(tmp_rand1,:);rand_tbl2(tmp_rand2,:)];

            % ensure every participant is selected
            if length(unique(rand_tbl.name)) == (length(unique(rand_tbl1.name)) + length(unique(rand_tbl2.name)))
                break
            end

        end

        y_data = rand_tbl.RTnorm;
        gms = processGMM(y_data, myFit.gaussN, myFit);

        mu_out(gi, p,:) = gms.mu;
        sigma_out(gi, p,:) = gms.sigma;
        weight_out(gi, p,:) = gms.weights;
        peak_out(gi,p,:) =  gms.peaks;

        gm_out.groups{gi, p} = gms;
    end
end
gm_out.mu_group = mu_out;
gm_out.sigma_group = sigma_out;
gm_out.w_group = weight_out;
gm_out.peak_group = peak_out;

[mu_out,sigma_out,weight_out,peak_out] = deal(nan(length(unique_groups),com_pN,myFit.gaussN));% how many comparsions

compPhase = {[1 2],[2 3],[1 3]};
for gi = 1:length(unique_groups)
    for com_p = 1:com_pN

        % select half data from each phase
        while 1
            rand_tbl1 = gtbl(gtbl.group==unique_groups(gi) & ismember(gtbl.phase,compPhase{com_p}(1)), :);
            rand_tbl2 = gtbl(gtbl.group==unique_groups(gi) & ismember(gtbl.phase,compPhase{com_p}(2)), :);

            tmp_rand1 = randperm(height(rand_tbl1),round(height(rand_tbl1)*0.5)); % select half of the data
            tmp_rand2 = randperm(height(rand_tbl2),round(height(rand_tbl2)*0.5)); % select half of the data

            rand_tbl = [rand_tbl1(tmp_rand1,:);rand_tbl2(tmp_rand2,:)];

            % ensure every participant is selected
            if length(unique(rand_tbl.name)) == length(unique(rand_tbl1.name))
                break
            end
        end

        y_data = rand_tbl.RTnorm;
        gms = processGMM(y_data, myFit.gaussN, myFit);

        mu_out(gi,com_p,:) = gms.mu;
        sigma_out(gi,com_p,:) = gms.sigma;
        weight_out(gi,com_p,:) = gms.weights;
        peak_out(gi,com_p,:) = gms.peaks;
        gm_out.phase{gi, com_p} = gms;
    end
end
gm_out.mu_phase = mu_out;
gm_out.sigma_phase = sigma_out;
gm_out.w_phase = weight_out;
gm_out.peak_phase = peak_out;
