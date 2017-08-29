function quantilenorm_init(num_cores)
    concurrency_gpus = 4;
    for i = 1:gpuDeviceCount
        semaphore(['/g' num2str(i)],'unlink');
        semaphore(['/g' num2str(i)],'open',concurrency_gpus);
    end
    for i = 1:length(num_cores)
        semaphore(['/c' num2str(i)],'unlink');
        semaphore(['/c' num2str(i)],'open',num_cores(i));
    end
end

