function registrationTest(registeredImagesDir, registeredImagesDirCompare, num_rounds);

keys_total = reporting_registration([], registeredImagesDir, num_rounds);
keys_total_compare = reporting_registration([], registeredImagesDirCompare, num_rounds);

assert(isequal(keys_total, keys_total_compare))
disp('Success, all number of keys match between rounds');
end
