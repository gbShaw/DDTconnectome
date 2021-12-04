function [y_predicted]= prediction_elastic_net(all_vcts,behav,covariables,alpha,thresh)

no_sub =length(behav);

% generate 10-fold indices
indices = crossvalind('Kfold',1:no_sub,10);  % matlab2020 supported

y_predicted=zeros(no_sub,1);

fprintf('Cross validation ')
for ki = 1:10
    fprintf('>')
    testInd = (indices == ki); trainInd = ~testInd;

    train_vcts   = all_vcts(trainInd,:) ;   
    train_behav  = behav(trainInd) ;


    train_covariables = covariables(trainInd ,: ) ;                        
    test_vct          = all_vcts(testInd,:);
    
    % feature selection    
     [r_mat,p_mat] = partialcorr (train_vcts,train_behav ,train_covariables ) ;


        

    selected_edges = p_mat < thresh;


   

    % build model on TRAIN subs  and report predicted behaviour  
    [B,FitInfo] = lasso(train_vcts(: , selected_edges) ,train_behav,'CV', 10 ,'Alpha',alpha  );

    b  = B(:,FitInfo.IndexMinMSE);
    b0 = FitInfo.Intercept(FitInfo.IndexMinMSE);

    FitInfo.b = b;
    FitInfo.b0 = b0;
    FitInfo.allBisZeros= logical(sum(b)) ;

    y_predicted(testInd) = sum(test_vct(:,selected_edges)' .* b , 1) + b0 ; 

end
fprintf('\n')


