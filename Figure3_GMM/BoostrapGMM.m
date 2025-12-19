function gm_out = BoostrapGMM(gtbl, myFit)
% Function to resample data and fit Gaussian Mixture Model (GMM) for each group and phase
% gtbl: Table with 'group', 'RTnorm', and 'phase' columns
% myFit: Structure containing fitting parameters (binN4group_perpart, norm, etc.)

unique_groups = unique(gtbl.group); % Get unique groups
gm_out = struct;
[mu_out,sigma_out,weight_out,peak_out] = deal(nan(length(unique_groups),3,myFit.gaussN));% how many comparsions

for gi = 1:length(unique_groups)

    for p = 1:3

        rand_tbl = gtbl(gtbl.group==unique_groups(gi) & gtbl.phase ==p, :);
        tmp_rand = randi(height(rand_tbl),height(rand_tbl),1); % with replacement
        rand_tbl = rand_tbl(tmp_rand,:);

        y_phase = rand_tbl.RTnorm;
        gms = processGMM(y_phase, myFit.gaussN, myFit);

        mu_out(gi, p,:) = gms.mu;

        sigma_out(gi, p,:) = gms.sigma;
        weight_out(gi, p,:) = gms.weights;
        peak_out(gi,p,:) =  gms.peaks;
        gm_out.groups{gi, p} = gms;

    end
    gm_out.mu = mu_out;
    gm_out.sigma = sigma_out;
    gm_out.w = weight_out;
    gm_out.peak = peak_out;
end

