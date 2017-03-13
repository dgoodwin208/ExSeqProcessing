function [prob_distro,bin_means ] = createEmpDistributionFromVector( datavec,edges)
%CREATEEMPDISTRIBUTIONFROMVECTOR Create an empirical distribution from the
%given data. Later interpolation is currently up to the user

    [prob_distro,~] = histcounts(datavec,edges);
    prob_distro = prob_distro/sum(prob_distro);
    bin_means = zeros(size(prob_distro));
    for x = 1:length(edges)-1
        bin_means(x) = mean(edges([x,x+1]));
    end
end

