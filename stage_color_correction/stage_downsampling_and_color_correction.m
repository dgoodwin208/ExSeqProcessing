function stage_downsampling_and_color_correction()

    loadParameters;

    [ret,messages] = check_files_in_downsample_all();
    if ~ret
        run('downsample_all.m');
        [ret,messages] = check_files_in_downsample_all();
        if ~ret
            for i = 1:length(messages)
                disp(messages{i})
            end
            return
        end
    else
        fprintf('already processed downsample_all\n');
    end

    [ret,messages] = check_files_in_color_correction();
    if ~ret
        
	%TODO: integrate this properly once it's worked on a few fields of view
	for i=1:params.NUM_ROUNDS
	    colorcorrection_3DWholeImage(i);
	end 
	%if params.USE_GPU_CUDA
        %    colorcorrection_3D_cuda();
        %else
        %    for i = 1:params.NUM_ROUNDS
        %        colorcorrection_3D(i);
        %    end
        %end
        
	[ret,messages] = check_files_in_color_correction();
        if ~ret
            for i = 1:length(messages)
                disp(messages{i})
            end
            return
        end
    else
        fprintf('already processed colorcollrection_3D\n');
    end

    [ret,messages] = check_files_in_downsample_apply();
    if ret
        fprintf('already processed downsample_apply\n');
        fprintf('[DONE]\n');
        return
    end

    make_links_in_color_correction_dir();

    run('downsample_applycolorshiftstofullres.m');
    [ret,messages] = check_files_in_downsample_apply();
    if ret
        fprintf('[DONE]\n');
    else
        for i = 1:length(messages)
            disp(messages{i})
        end
    end
end
