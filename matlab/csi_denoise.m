function results = csi_denoise (csi_info)
    
    %butterworth LPF
    Fs = 500;
    %The cutoff frequency is 30Hz
    [T1 , T2] = butter(6 , 30/(Fs/2) , 'low');

%   [h,f]=freqz(T1,T2,'whole',500);  
%   f=(0:length(f)-1*Fs/length(f));  
%   plot(f(1:length(f)/2),abs(h(1:length(f)/2)));
%   xlabel('Frequency/Hz')
%   ylabel('Amplitude')
%   title('Frequency Response of 6-order Butterworth LPF')


    csi_butter = filter(T1, T2, csi_info, [], 2);
    csi_butter = csi_butter(:, 100:end);
    win_len = Fs;
    

    %Remove the moving average of CSI
    csi_mov_mean = movmean(csi_butter, win_len, 2);
    results  = csi_butter - csi_mov_mean;
 

end

    