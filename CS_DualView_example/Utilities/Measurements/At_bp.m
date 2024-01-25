function x = At_bp(b,OMEGA,P_image,P_block,image_height,image_width,block_height,block_width,Phi)    
    n=length(P_image)/(block_width*block_height);
    [M,N] = size(Phi);
    fx=zeros(n,M);
    if iscell(OMEGA)
        OMEGA=cell2mat(OMEGA);
    end
    if iscell(b)
        b=cell2mat(b);
    end
    fx(OMEGA)=b;
   
    fx=fx';
    x=zeros(N,n);
    x(P_block,:)=Phi'*fx;
    x=col2im(x,[block_height,block_width],[image_height,image_width],'distinct');
    x=x(:);
    x(P_image)=x;
    
end