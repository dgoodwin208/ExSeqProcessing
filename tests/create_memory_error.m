function create_memory_error()
    test = zeros(100000);
    hold_it = [];
    stepper = 1;
    while(stepper < 1000)
        temp = fft(test);
        hold_it = [hold_it temp];
        stepper = stepper + 1;
    end
end
