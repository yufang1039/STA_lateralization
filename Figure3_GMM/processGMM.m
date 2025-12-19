function gms = processGMM(y_data, gaussN, myFit)
gm = fitgmdist(y_data, gaussN, 'Options', myFit.options, 'Start', myFit.Start);

% sort the clusters based on mean RT
[mu_sorted, idx] = sort(gm.mu);
sigma_sorted = squeeze(gm.Sigma(idx));
weights = gm.ComponentProportion(idx)';
gms.peaks = weights .* normpdf(mu_sorted, mu_sorted, sqrt(sigma_sorted));
gms.post_p = posterior(gm, y_data);
gms.mu = mu_sorted;
gms.weights = weights;
gms.sigma = sigma_sorted;
gms.AIC = gm.AIC;
gms.BIC = gm.BIC;

% making cluster 1 the faster component
if gm.mu(1) <gm.mu(2)
    gms.cluster = cluster(gm,y_data);
else
    gms.cluster = 3-cluster(gm,y_data);
end

%% AUC
mu1 = mu_sorted(1); sigma1 = sigma_sorted(1); w1 = weights(1);
mu2 = mu_sorted(2); sigma2 = sigma_sorted(2); w2 = weights(2);

% Weighted PDFs
f1 = @(x) w1 * normpdf(x, mu1, sigma1);
f2 = @(x) w2 * normpdf(x, mu2, sigma2);

% Find intersection points
diff_fun = @(x) f1(x) - f2(x);
x1 = fzero(diff_fun, mu1);
x2 = fzero(diff_fun, mu2);

% Ensure x1 < x2
x_low = min(x1, x2);
x_high = max(x1, x2);

% Non-overlapping area under f1
area_f1_left  = integral(@(x) f1(x), -Inf, x_low);
area_f1_right = integral(@(x) (f1(x) - f2(x)) .* (f1(x) > f2(x)), x_low, x_high);
area_f1 = area_f1_left + area_f1_right;

% Non-overlapping area under f2
area_f2_right = integral(@(x) f2(x), x_high, Inf);
area_f2_left = integral(@(x) (f2(x) - f1(x)) .* (f2(x) > f1(x)), x_low, x_high);
area_f2 = area_f2_left + area_f2_right;

gms.auc(1) = area_f1; % cluster1
gms.auc(2) = area_f2; % cluster2
gms.auc(3) = 1 - (area_f1 + area_f2); % overlapping area


% Gaussian CDF handle
Phi = @(z) 0.5*(1+erf(z./sqrt(2)));

p_1sd = Phi(1) - Phi(-1);   % ≈0.6827

% (A) Component-only AUCs (constant 0.6827 * weight)
auc_comp1 = w1 * p_1sd;
auc_comp2 = w2 * p_1sd;
gms.auc_1sd_component = [auc_comp1, auc_comp2];

% (B) Mixture AUCs within each component’s ±1σ window
a1 = mu1 - sigma1; b1 = mu1 + sigma1;
a2 = mu2 - sigma2; b2 = mu2 + sigma2;

% helper: mass of N(mu,s^2) in [a,b]
norm_mass = @(a,b,mu,s) Phi((b-mu)/s) - Phi((a-mu)/s);

auc_mix1 = w1*norm_mass(a1,b1,mu1,sigma1) + w2*norm_mass(a1,b1,mu2,sigma2);
auc_mix2 = w1*norm_mass(a2,b2,mu1,sigma1) + w2*norm_mass(a2,b2,mu2,sigma2);

gms.auc_1sd_mixture = [auc_mix1, auc_mix2];

% ----- AUC within ±2 SD -----
m = 0.5;
p_2sd = normcdf(m) - normcdf(-m);   % ~0.9545

% (A) Component-only
auc2_comp1 = w1 * p_2sd;
auc2_comp2 = w2 * p_2sd;
gms.auc_2sd_component = [auc2_comp1, auc2_comp2];

% (B) Mixture AUCs within ±2σ windows
a1 = mu1 - m*sigma1; b1 = mu1 + m*sigma1;
a2 = mu2 - m*sigma2; b2 = mu2 + m*sigma2;
Phi = @(z) 0.5*(1+erf(z./sqrt(2)));
norm_mass = @(a,b,mu,s) Phi((b-mu)/s) - Phi((a-mu)/s);

auc2_mix1 = w1*norm_mass(a1,b1,mu1,sigma1) + w2*norm_mass(a1,b1,mu2,sigma2);
auc2_mix2 = w1*norm_mass(a2,b2,mu1,sigma1) + w2*norm_mass(a2,b2,mu2,sigma2);
gms.auc_2sd_mixture = [auc2_mix1, auc2_mix2];

end