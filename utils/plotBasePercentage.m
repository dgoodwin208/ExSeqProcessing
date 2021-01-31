function [perc_base] = plotBasePercentage(transcripts, roundnumbers)
for base_idx = 1:4
    %     perc_base(:,base_idx) = sum(transcripts==base_idx,1)/size(transcripts,1);
    perc_base(:,base_idx) = sum(transcripts==base_idx,1)./sum(transcripts>0,1);
end
figure;
loadParameters;
if params.ISILLUMINA
    plotcolors = {'b','g','m','r'};
else %SOLiD coloring scheme
    plotcolors = {'b','g','m','r'};
end

plot(roundnumbers,perc_base(:,1)*100,plotcolors{1},'LineWidth',2); hold on;
plot(roundnumbers,perc_base(:,2)*100,plotcolors{2},'LineWidth',2)
plot(roundnumbers,perc_base(:,3)*100,plotcolors{3},'LineWidth',2)
plot(roundnumbers,perc_base(:,4)*100,plotcolors{4},'LineWidth',2); hold off;

if params.ISILLUMINA
    legend('Chan00 - G','Chan01 - T', 'Chan02 - A', 'Chan03 - C');
else %SOLiD coloring scheme
    legend('Chan00 - C','Chan01 - A', 'Chan02 - T', 'Chan03 - G');
end

title(sprintf('Percentage of each base across rounds for %i puncta',size(transcripts,1)));
xlabel('In situ sequencing round number');
ylabel('%');
xticks(roundnumbers)
end

