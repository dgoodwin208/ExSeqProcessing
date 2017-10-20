function quantilenorm_init(num_gpu_sem,num_cores)
    for i = 1:gpuDeviceCount
        semaphore(['/g' num2str(i)],'unlink');
        semaphore(['/g' num2str(i)],'open',num_gpu_sem);
    end
    for i = 1:length(num_cores)
        semaphore(['/c' num2str(i)],'unlink');
        semaphore(['/c' num2str(i)],'open',num_cores(i));
    end
end

