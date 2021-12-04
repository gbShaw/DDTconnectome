

clear 

%% parameter
alpha = 0.5;
thresh =  0.01;
n_repetition = 100;
n_permutation_it = 2000;

%% load data
load behav.mat          %% size:N*1
load FCmats.mat         %% size:M*M*N
load covariables.mat    %% size:N*C


%% prepare feature vector
nSub             = size(FCmats,3);
emptyMask        = FCmats(:,:,1)*0;
FCvcts           = reshape(FCmats, [], nSub)';

 % remove repeated FCs
feature_idx      = find(triu(ones(size(emptyMask)),1)); 
feature_delete   = ~(triu(ones(size(emptyMask)),1));
FCvcts(:,feature_delete) =[];



%% prediction 
r = zeros(n_repetition,1);
predicted_true = zeros(nSub,n_repetition);

parfor rp = 1: n_repetition

    predicted_true(:,rp) = prediction_elastic_net(FCvcts,behav,covariables,alpha,thresh);

end

r_true = corr( mean(predicted_true,2), behav)


%% permutation test
permutation_r    = zeros(n_permutation_it,1);
permutation_r(1) = r_true;
parfor it = 2: n_permutation_it

    new_behav         = behav(randperm(nSub));
    new_cor           = covariables(randperm(nSub),:);    
    
    predicted         = prediction_elastic_net(FCvcts,new_behav,new_cor,alpha,thresh);

    permutation_r(it) = corr(predicted, behav);
   
end


sorted_permutation_r  = sort(permutation_r,'descend') ;
position              = find(sorted_permutation_r==r_true);
        
permutation_pval      = position/n_permutation_it 








