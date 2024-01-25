function b = A_bp(x,OMEGA,P_image,P_block,image_height,image_width,block_height,block_width,Phi)

x=x(P_image);
x=reshape(x,[image_height,image_width]);
B=im2col(x,[block_height,block_width],'distinct');
B=B(P_block,:);

fx=Phi*B;
fx=fx';
if iscell(OMEGA)
    for i=1:length(OMEGA)
        b{i}=fx(OMEGA{i});
    end
else
    b=fx(OMEGA);
end

end