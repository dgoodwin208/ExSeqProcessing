function quantilenorm_init(num_gpu_sem,num_cores)
    for i = 1:gpuDeviceCount
        sem_name = sprintf('/%s.g%d',getenv('USER'),i);
        semaphore(sem_name,'unlink');
        semaphore(sem_name,'open',num_gpu_sem);
    end
    for i = 1:length(num_cores)
        sem_name = sprintf('/%s.c%d',getenv('USER'),i);
        semaphore(sem_name,'unlink');
        semaphore(sem_name,'open',num_cores(i));
    end
end

