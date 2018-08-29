function quantilenorm_final(num_core_sem)
    for i = 1:gpuDeviceCount
        sem_name = sprintf('/%s.g%d',getenv('USER'),i);
        semaphore(sem_name,'unlink');
    end
    for i = 1:num_core_sem
        sem_name = sprintf('/%s.c%d',getenv('USER'),i);
        semaphore(sem_name,'unlink');
    end
end

