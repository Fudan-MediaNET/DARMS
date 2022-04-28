function extract_segment(data_path, movement, save_path, segment_length)
    data_dir  = dir([data_path movement '_*.mat']);
    for num_file=1:length(data_dir)
        read_name = [data_path data_dir(num_file).name];
        load(read_name);
        len_csi = size(processed_csi_info, 2);
    
        variance = zeros(180, len_csi-segment_length);
        for iter=1:len_csi-segment_length
            variance(:, iter)=var(processed_csi_info(: ,iter:iter+segment_length), 0, 2);
        end
        [~, max_col] = max(variance, [], 2);
        center_col = round(mean(max_col)) + segment_length/2;
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

