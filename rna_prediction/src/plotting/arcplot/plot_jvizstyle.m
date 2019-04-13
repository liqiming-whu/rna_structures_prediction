function plot_jvizstyle(mfe_ct, native_ct, mfe_counts, native_counts)
    
    length = size(mfe_ct,1);
    
    pairings_mfe = mfe_ct(:,3);
    pairings_native = native_ct(:,3);
    
    only_mfe = pairings_mfe;
    only_mfe(pairings_mfe == pairings_native) = 0;
    only_native = pairings_native; 
    only_native(pairings_mfe == pairings_native) = 0;

    common = pairings_native; 
    common(pairings_mfe ~= pairings_native) = 0; 
    
%     subplot(1,3,1)
     h = figure; 
     color1 = arcplot(only_mfe, ones(size(only_native))*0.5, 90, true, lines(16), true)
     hold on;
     color2 = arcplot(only_native, ones(size(only_mfe))*1.0, 90, true, lines(16),false)
     color3 = arcplot(common, ones(size(only_native))*0.6, 90, true, lines(16),false)
     colorbar('off')
     rectangle('Position', [-1.2, -1.35,0.3,0.1], 'FaceColor', color1, 'EdgeColor', color1); 
     rectangle('Position', [-0.15, -1.35,.3,0.1], 'FaceColor', color2, 'EdgeColor', color2); 
     rectangle('Position', [0.9, -1.35,0.3,0.1], 'FaceColor', color3, 'EdgeColor', color3); 
     text(-1.25, -1.42, 'MFE only','FontSize', 12);
     text(-0.2, -1.42, 'Native only', 'FontSize', 12);
     text(0.85, -1.42, 'Common','FontSize', 12);
     %pos = get(gca, 'CameraPosition');
     %set(gca, 'CameraPosition', pos + [0 0.05 0]);
     %get(h,'Position')
     %legend('MFE only', 'Native only', 'Common', 'Location', 'EastOutside')
     %title('Comparison of MFE and native structures')
    % colorbar('off')
     hold off;

    h2 = figure;
    %subplot(1,3,2)
    arcplot(pairings_mfe, mfe_counts, 90, true);
    %pos = get(h2,'PaperPosition')
    %set(h, 'PaperPosition', pos)
    %get(h, 'PaperPosition')
   % title('Frequency of pairs in data-directed structures (MFE)')
    colorbar('off')
    
%     subplot(1,3,3)
figure;
     arcplot(pairings_native, native_counts,90, true);
    % title('Frequency of pairs in data-directed structures (native)')
    %colorbar('off')

end