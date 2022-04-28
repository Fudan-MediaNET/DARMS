clc;
clear;


data_path = '../raw_data/';
tmp_path = '../tmp/';
save_path = '../processed_csi/';

movement = {'empty', 'sit', 'walk', 'stand', 'stooll', 'stoolr', 'stoolf'};


%Denoising the raw CSI data
for num_mov = 1:length(movement)
    csi_preprocess(data_path, movement{num_mov}, tmp_path);
end



%Extracting the CSI segment that contains movements
for num_mov = 1:length(movement)
    extract_segment(tmp_path, movement{num_mov}, save_path, 800);
end



