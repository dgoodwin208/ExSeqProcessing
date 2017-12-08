function tests = SWITCH_tile_processingTest
    tests = functiontests(localfunctions);
end

function setup(testCase)

    testCase.TestData.tempDir = '/mp/nas1/share/tmp/testworks';
    if ~exist(testCase.TestData.tempDir)
        mkdir(testCase.TestData.tempDir)
    end

    img={};
    chans = 1
    for i = 1:chans
        img{i}=rand(10,10,80);
    end
    img{1}(2,1,1)=0.1;
    img{1}(4,1,1)=0.1;
    for i = 1:chans
        save3DTif(img{i},fullfile(testCase.TestData.tempDir,sprintf('test_%d.tif',i)));
    end

end

function teardown(testCase)
    quantilenorm_final(1);

    rmdir(testCase.TestData.tempDir,'s');
end

% utility functions
function image = load_binary_image(outputdir,image_fname,image_height,image_width)
    fid = fopen(fullfile(outputdir,image_fname),'r');
    count = 1;
    while ~feof(fid)
        sub_image = fread(fid,[2,image_height*image_width],'double');
        %sub_image = fread(fid,[image_height,image_width],'double');
        if ~isempty(sub_image)
            image(:,:,count) = reshape(sub_image(2,:),[image_height image_width]);
            %image(:,:,count) = sub_image;
            count = count + 1;
        end
    end
    fclose(fid);
end

% normal test cases
function testQuantilenorm(testCase)

    img={};
    chans = 1
    for i = 1:chans
        img{i}=load_binary_image(fullfile(testCase.TestData.tempDir, sprintf('test_%d.tif', i)));
    end

    keys=SWITCH_tile_processing(test);

end

