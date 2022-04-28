function extract_segment(data_path, movement, save_path, segment_length)
    data_dir  = dir([data_path movement '_*.mat']);
    for num_file=1:length(data_dir)
        read_name = [data_path data_dir(num_file).name];
        load(read_name);
        len_csi = size(processed_csi_info, 2);
    
        csi_var = movvar(processed_csi_info, segment_length, 0, 2);
        sum_csi_var = sum(csi_var);

        mov_sum = movsum(sum_csi_var, segment_length);
        [~, center_col] = max(mov_sum);

        first = center_col - (segment_length/2-1);
        last = center_col + segment_length/2;
        if first <= 0
            first = 1;
            last = segment_length;
        elseif last > len_csi
            first = len_csi - segment_length + 1;
            last = len_csi;
        end
        csi_segment = processed_csi_info(:, first:last);
             
        write_name = [save_path  movement '_' num2str(num_file)  '.mat'];
        save(write_name, 'csi_segment');
        
        fprintf('run to %s_%d\n', movement, num_file);
    end
end

