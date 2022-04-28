function csi_preprocess( data_path,movement,save_path )
    data_dir_a  = dir([data_path movement '_a_*']);
    data_dir_b  = dir([data_path movement '_b_*']);
    for num_file=1:length(data_dir_a)
        %Reading the CSI of receiver A
        read_name_a = data_dir_a(num_file).name;
        csi_trace_a = read_bf_file([data_path, read_name_a]);
        data_length_a = size(csi_trace_a,1);
        
        %Reading the CSI of receiver B
        read_name_b = data_dir_b(num_file).name;
        csi_trace_b = read_bf_file([data_path, read_name_b]);
        data_length_b = size(csi_trace_b,1);    
        
        data_length = min(data_length_a, data_length_b);
        csi_info = zeros(180,data_length);
        for col_num = 1:data_length
            temp_a = get_scaled_csi(csi_trace_a{col_num});
            temp_b = get_scaled_csi(csi_trace_b{col_num});
            for rx_num = 1:3
                for sub_num = 1:30
                    csi_info(sub_num + 30*(rx_num-1),col_num) = temp_a(1,rx_num,sub_num).*conj(temp_a(1,rx_num,sub_num)); 
                    csi_info(90 + sub_num + 30*(rx_num-1),col_num) = temp_b(1,rx_num,sub_num).*conj(temp_b(1,rx_num,sub_num));
                end
            end
        end
        
        %Denoising the CSI
        processed_csi_info = csi_denoise(csi_info);
        fprintf('run to %s_%d\n', movement, num_file);
        write_name = [save_path  movement '_' num2str(num_file)  '.mat'];
        save(write_name , 'processed_csi_info');
    end
end

