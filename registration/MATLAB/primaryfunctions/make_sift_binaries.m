function [] = make_sift_binaries(img, map, image_size, sift_params, N)

    assert(all(size(map) == size(img)));
    f = fopen('test_map.bin', 'w');
    fwrite(f, image_size(1), 'uint32');
    fwrite(f, image_size(2), 'uint32');
    fwrite(f, image_size(3), 'uint32');
    fwrite(f, double(map), 'double');
    fclose(f);
    fprintf('Saved test_map with %d real keypoints\n', N);

    f = fopen('test_img.bin', 'w');
    fwrite(f, image_size(1), 'uint32');
    fwrite(f, image_size(2), 'uint32');
    fwrite(f, image_size(3), 'uint32');
    fwrite(f, double(img), 'double');
    fclose(f);
    fprintf('Saved test_img \n');

    f = fopen('fv_centers.bin', 'w');
    fwrite(f, sift_params.fv_centers_len, 'uint32');
    fwrite(f, double(sift_params.fv_centers), 'double');
    fclose(f);
    fprintf('Saved fv \n');

end
