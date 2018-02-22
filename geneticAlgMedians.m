close all; clear all; clc;



load('medianStats')
rng('shuffle');

strtVec = zeros(size(stats3Agg,2),1);
stats3Agg(:,32:34) = []; %these perfectly predict (wins, losses, wins-losses)

initPop = [];
for i = 1:size(stats3Agg,2)-8 
chngVec = strtVec;
chngVec(0+i:8+i) = 1;
initPop = [initPop chngVec];
end

% for i = 1:7
% addVec = strtVec; addVec(tester.(sprintf('t%d',i))) = 1;
% initPop = [initPop addVec];
% end

for i = 1:100
addRndVec = strtVec; r = randi(size(stats3Agg,2),9,1);
while length(unique(r)) ~= 9
    r = randi(size(stats3Agg,2),9,1);
end
addRndVec(r) = 1;
initPop = [initPop addRndVec];
end 

%x(1, [2 4 6 9])

%y2(y2==0)=2;
for iter = 1:100
    costFun = [];
for i = 1:size(initPop,2)
y = [y2' ones(length(y2),1)];
%y = y2';
x = stats3Agg(:,initPop(:,i)==1);
[B, dev, stats] = glmfit(x,y, 'binomial', 'link', 'logit','constant','on');
YHAT = glmval(B,x,'logit','constant','on');
YHAT(YHAT(:,1) >= 0.5,1) = 1; YHAT(YHAT(:,1) < 0.5,1) = 0; %coded for which team wins with 0.5 p as a cutoff
% accuracy = sum(y2'==YHAT)/length(y2);
accuracy = sum(y(:,1)==YHAT)/length(y);
multicolin = abs(corr(x,'type','spearman'));
corr_penalty = sum(multicolin(multicolin>0.5 & multicolin<1));

 %costFun(i) = accuracy./var(abs(B(2:end))); %9 average predictors cost function
 %costFun(i) = accuracy./(var(abs(B(2:end))) + (corr_penalty)); %9 average predictors cost function
%costFun(i) = accuracy;
B2 = sort(B(2:end),'descend'); costFun(i) = (accuracy+sum(abs(B(1:3))))./(var(abs(B(1:3)))+sum(abs(B(4:9)))) - corr_penalty; %3+6 predictors cost function
%B2 = sort(B(2:end),'descend'); costFun(i) = accuracy./(var(abs(B2(1:3)))+(sum(abs(B2(4:9))))); %3+6 predictors cost function

% B2 = abs(B); B2 = sort(B2(2:end),'ascend');  %TTB predictors
% costFun(i) = (B2(9)/(B2(8)+B2(7))).*(B2(8)/(B2(7)+B2(6))).*(B2(7)/(B2(6)+B2(5))).*(B2(6)/(B2(5)+B2(4))).*(B2(5)/(B2(4)+B2(3))).*(B2(4)/(B2(3)+B2(2))).*(B2(3)/(B2(2)+B2(1))).*(B2(2)/(B2(1))) %- corr_penalty; %ttb
end
      [s,Ind] = sort(costFun,'descend');
      initPop_temp = initPop;
      
 for i = 1:100
 p=fliplr((1:size(initPop,2))./sum(1:size(initPop,2)));
 pd = makedist('Multinomial','Probabilities',p);
 
 parent1 = Ind(random(pd));  parent2 = Ind(random(pd));
 crossover = randi(8);
 parent1_preds = find(initPop(:,parent1));  parent2_preds = find(initPop(:,parent2));
 child_preds = [parent1_preds(1:crossover);  parent2_preds(crossover+1:9)];
 child_preds = unique(child_preds);
 
while length(unique(child_preds)) < 9
     mutation = randi(size(stats3Agg,2));
     child_preds = [child_preds; mutation];
 end
 
 if randi(5) == 5
     mutation = randi(size(stats3Agg,2)); 
     while ismember(mutation,child_preds) == 1
         mutation = randi(size(stats3Agg,2));
     end
     child_preds(randi(9)) = mutation;
 end
 
 addVec = strtVec; addVec(child_preds) = 1;
 initPop_temp = [initPop_temp addVec];
 end

 initPop = initPop_temp(:,end-99:end);
end

x = stats3Agg(:,initPop(:,1)==1);
[B, dev, stats] = glmfit(x,y, 'binomial', 'link', 'logit','constant','on');
YHAT = glmval(B,x,'logit','constant','on');
YHAT(YHAT(:,1) >= 0.5,1) = 1; YHAT(YHAT(:,1) < 0.5,1) = 0;
accuracy = sum(y(:,1)==YHAT)/length(y);
bar(abs(B(2:end)))
find(initPop(:,1))