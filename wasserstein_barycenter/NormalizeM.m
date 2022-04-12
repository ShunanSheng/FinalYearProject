function M_cell = NormalizeM(M_cell)
    % input M x 1 cell, output normalized cell
    M_all = cell2mat(M_cell);
    med = median(M_all, 'all');
    s = max(size(M_cell));
    for i = 1:s
        M_cell{i} = M_cell{i}./med;
    end
end