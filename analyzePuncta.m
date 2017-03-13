%Produce the set of puncta using the getPuncta.m file
loadParameters;
% puncta_directory = '/Users/Goody/Neuro/ExSeq/rajlab/splintr1/';
load(fullfile(params.rajlabDirectory ,'puncta_allexp.mat'));

%load sample image for reference
img = load3DTif(fullfile(params.rajlabDirectory ,'alexa001.tiff'));

% NUM_ROUNDS = 3;
%% make a quick scatter plot
figure; hold on;

for exp_idx = 1:params.NUM_ROUNDS
   locs = puncta{exp_idx};
   scatter(locs(:,1),locs(:,2),'.');
end
xlim([1,size(img,2)]);
ylim([1,size(img,1)]);
title('Raw, unfiltered RajLab puncta candidates (displayed in redundant color)');
hold off;

%% Remove any redundant points from the rajlab code

for exp_idx = 1:params.NUM_ROUNDS
   deduped = removeRedundantPuncta(puncta{exp_idx});
   fprintf('Round #%i, Removed %i redundant puncta of %i candidates\n',...
       exp_idx,...
       size(puncta{exp_idx},1)-size(deduped,1),...
       size(puncta{exp_idx},1));
   puncta{exp_idx} = deduped;
end


%% Make histogram of neighbors around the reference, currently set as #1

%Note: REF_ROUND as a variable is not fully implemented in the code
%Specifically in the nested for loops below in the 2:12 hardcodes
REF_ROUND = 1;
epsilon = 1:10;

puncta_ref = puncta{REF_ROUND};
%The buckets of are of dimension
%[length(reference number), number of rounds -1, number of neighbors]
buckets = zeros(size(puncta{1},1),params.NUM_ROUNDS-1,length(epsilon));


for puncta_idx = 1:size(puncta_ref,1)
    
    %first do a rough filtering of the dataset to get only the regions
    %around a puncta
    puncta_location = puncta_ref(puncta_idx,:);
    
    %generate a 2*max(epsilon)+1 size window around the puncta of interest
    y_min = puncta_location(1) - max(epsilon);
    y_max = puncta_location(1) + max(epsilon);
    x_min = puncta_location(2) - max(epsilon);
    x_max = puncta_location(2) + max(epsilon);
    z_min = puncta_location(3) - max(epsilon);
    z_max = puncta_location(3) + max(epsilon);
    
    candidate_puncta_neighbors = [];
    for other_rd_idx = 2:params.NUM_ROUNDS
        otherpuncta_locations = puncta{other_rd_idx};
        
        y_candidates = otherpuncta_locations(:,1);
        y_indices = y_candidates>=y_min & y_candidates<=y_max;
        
        x_candidates = otherpuncta_locations(:,2);
        x_indices = x_candidates>=x_min & x_candidates<= x_max;
        
        z_candidates = otherpuncta_locations(:,3);
        z_indices = z_candidates>=z_min & z_candidates <= z_max;
        
        %Keep the indices that satisfy all three dimension constraints
        otherpuncta_indices = y_indices & x_indices & z_indices;
        otherpuncta_neighbors = otherpuncta_locations(otherpuncta_indices,:);
        
        if length(otherpuncta_neighbors)>0
            %calculate all the distances 
            c = [(otherpuncta_neighbors(:,1) - puncta_location(1)),...
                 (otherpuncta_neighbors(:,2) - puncta_location(2)), ...
                 (otherpuncta_neighbors(:,3) - puncta_location(3))];
            distances = diag(sqrt(c*c'));
            
            %get the minimum distance, that determines the bucket to fill
            %if you get one distance, you get all the greater distances too
            buckets(puncta_idx,other_rd_idx,floor(min(distances))+1:end) = 1;
        end
    end
    
    if mod(puncta_idx,200)==0
        fprintf('Processing puncta_idx %i/%i\n',puncta_idx,size(puncta{1},1));
    end
end

%% collapse the buckets into a 2d vector then histogram
%Want a matrix that is just 2D:
%[episolons],[number of rounds that] -> number of puncta

%test one epsilon

rounds_counts = zeros(size(buckets,3),params.NUM_ROUNDS-1);

%For a specific
for r=1:params.NUM_ROUNDS-1
    for e = 1:length(epsilon)
        t = squeeze(buckets(:,:,e));
        rounds_counts(e,r) = sum(sum(t,2)==r);
    end
end
figure; imagesc(rounds_counts)

%%

figure; 
hold on;
colors = {'ro-','go-','bo-','co-','mo-','ro--','go--','bo--','co--','mo--'};
for e = 1:length(epsilon)
%     plot(rounds_counts(e,:),'o-','LineWidth',2)
    plot(rounds_counts(e,:),colors{e},'LineWidth',2)
end
xlabel('Number of rounds within epsilon');
ylabel(sprintf('Number of puncta (%i original candidates)',size(puncta_ref,1)));
legend('1','2','3','4','5','6','7','8','9','10','Location','northwest');
title(sprintf('Number of puncta that are within an epsilon across number of rounds\n%s',params.rajlabDirectory ));

%% Finally, get a list of all puncta that we would use for later analysis

%Only use puncta that are present in THRESHOLD number of rounds at specific 
%epsilon

puncta_votes = zeros(1,size(puncta_ref,1));

for puncta_idx = 1:size(puncta_ref,1)
    
    %first do a rough filtering of the dataset to get only the regions
    %around a puncta
    puncta_location = puncta_ref(puncta_idx,:);
    
    %generate a 2*max(epsilon)+1 size window around the puncta of interest
    y_min = puncta_location(1) - params.EPSILON_TARGET;
    y_max = puncta_location(1) + params.EPSILON_TARGET;
    x_min = puncta_location(2) - params.EPSILON_TARGET;
    x_max = puncta_location(2) + params.EPSILON_TARGET;
    z_min = puncta_location(3) - params.EPSILON_TARGET;
    z_max = puncta_location(3) + params.EPSILON_TARGET;
    
    candidate_puncta_neighbors = [];
    
    for other_rd_idx = 2:params.NUM_ROUNDS
        otherpuncta_locations = puncta{other_rd_idx};
        
        y_candidates = otherpuncta_locations(:,1);
        y_indices = y_candidates>=y_min & y_candidates<=y_max;
        
        x_candidates = otherpuncta_locations(:,2);
        x_indices = x_candidates>=x_min & x_candidates<= x_max;
        
        z_candidates = otherpuncta_locations(:,3);
        z_indices = z_candidates>=z_min & z_candidates <= z_max;
        
        %Keep the indices that satisfy all three dimension constraints
        otherpuncta_indices = y_indices & x_indices & z_indices;
        otherpuncta_neighbors = otherpuncta_locations(otherpuncta_indices,:);
        
        %If there is a point within the epsilon_target, give it a vote
        if length(otherpuncta_neighbors)>0
           puncta_votes(puncta_idx)= puncta_votes(puncta_idx)+1;
        end
    end
    
    if mod(puncta_idx,1000)==0
        fprintf('Processing puncta_idx %i/%i\n',puncta_idx,size(puncta{1},1));
    end
end
%% 
figure;
% Plot all the puncta in blue for the reference channel (currently #1)
scatter(puncta_ref(:,1),puncta_ref(:,2),'b.');
xlim([1,size(img,2)]);
ylim([1,size(img,1)]);
hold on;
%Plot all the puncta that had very few votes
locs = puncta_ref(puncta_votes<params.THRESHOLD/2,:);
scatter(locs(:,1),locs(:,2),'r.');
%plot all the puncta that satisfy the voting threshold
locs = puncta_ref(puncta_votes>=params.THRESHOLD,:);
scatter(locs(:,1),locs(:,2),'g.');
legend('Reference rnd puncta','Too few','Passed');
hold off;

%% Save the puncta and the parameters they were made at
puncta_filtered = puncta_ref(puncta_votes>=params.THRESHOLD,:);
save(fullfile(params.rajlabDirectory ,'puncta_filtered.mat'),'puncta_filtered');
