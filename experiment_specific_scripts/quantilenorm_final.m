function quantilenorm_final(num_core_sem)
    for i = 1:gpuDeviceCount
        semaphore(['/g' num2str(i)],'unlink');
    end
    for i = 1:num_core_sem
        semaphore(['/c' num2str(i)],'unlink');
    end
end

