

% this is pulled from Dafna's stackoverflow reply at
% https://stackoverflow.com/questions/44591037/speed-up-calculation-of-maximum-of-normxcorr2
% it is to be translated into python



pkg load image

function [N] = naive_corr(pat,img)
    [n,m] = size(img);
    [np,mp] = size(pat);
    N = zeros(n-np+1,m-mp+1);
    for i = 1:n-np+1
        for j = 1:m-mp+1
            N(i,j) = sum(dot(pat,img(i:i+np-1,j:j+mp-1)));
        end
    end
end


%w_arr the array of coefficients for the boxes
%box_arr of size [k,4] where k is the number boxes, each box represented by
%4 something ...
function [C] = box_corr2(img, box_arr, w_arr, n_p, m_p)

    % construct integral image + zeros pad (for boundary problems)
    % B = cumsum(A,dim) returns the cumulative sum of the elements along dimension dim. For example, if A is a matrix, then cumsum(A,2) returns the cumulative sum of each row.
    I = cumsum(cumsum(img,2),1);
    I = [zeros(1,size(I,2)+2); [zeros(size(I,1),1) I zeros(size(I,1),1)]; zeros(1,size(I,2)+2)];

    % initialize result matrix
    [n,m] = size(img);
    C = zeros(n-n_p+1,m-m_p+1);
    %C = zeros(n,m);
    % C
    jump_x = 1;
    jump_y = 1;

    x_start = ceil(n_p/2);
    x_end = n-x_start+mod(n_p,2);
    x_span = x_start:jump_x:x_end;

    y_start = ceil(m_p/2);
    y_end = m-y_start+mod(m_p,2);
    y_span = y_start:jump_y:y_end;

    arr_a = box_arr(:,1) - x_start;
    arr_b = box_arr(:,2) - x_start+1;
    arr_c = box_arr(:,3) - y_start;
    arr_d = box_arr(:,4) - y_start+1;

    % cumulate box responses
    k = size(box_arr,1); % == numel(w_arr)

    for i = 1:k
        a = arr_a(i);
        b = arr_b(i);
        c = arr_c(i);
        d = arr_d(i);

        C = C ...
            + w_arr(i) * ( I(x_span+b,y_span+d) ...
                           - I(x_span+b,y_span+c) ...
                           - I(x_span+a,y_span+d) ...
                           + I(x_span+a,y_span+c) );
    end

end



function [NCC]  = naive_normxcorr2(temp,img)

    [n_p,m_p]=size(temp);

    M = n_p*m_p;

    % compute template mean & std
    temp_mean = mean(temp(:));
    temp = temp - temp_mean;

    temp_std = sqrt(sum(temp(:).^2)/M);

    % compute windows' mean & std
    wins_mean =  box_corr2(img,[1,n_p,1,m_p],1/M,  n_p,m_p);
    wins_mean2 = box_corr2(img.^2,[1,n_p,1,m_p],1/M,n_p,m_p);


    wins_std = real(sqrt(wins_mean2 - wins_mean.^2));
    NCC_naive = naive_corr(temp,img);

    NCC = NCC_naive ./ (M .* temp_std .* wins_std);
end



n = 170;

particle_1=rand(54,54,n);
particle_2=rand(56,56,n);

particle_1
"particle_1"

[n_p1,m_p1,c_p1]=size(particle_1);
[n_p2,m_p2,c_p2]=size(particle_2);

L1 = zeros(n,1);
L2 = zeros(n,1);


% tic
% for i=1:n
%     C1=normxcorr2(particle_1(:,:,i),particle_2(:,:,i));
%
%     C1_unpadded = C1(n_p1:n_p2 , m_p1:m_p2);
%     L1(i)=max(C1_unpadded(:));
%
% end
% toc


tic
for i=1:n

    C2=naive_normxcorr2(particle_1(:,:,i),particle_2(:,:,i));
    L2(i)=max(C2(:));
end
toc
