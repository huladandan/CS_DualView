function [A, At, measure] = Measure_matrix_create(measure, seed)
randn('state',seed);
rand('state',seed);
switch measure.model
    
    case  'Bernoulli'
        k = ceil(length(measure.OMEGA)/(measure.image_height*measure.image_width/(measure.block_width*measure.block_height)));
        p = 0.5;
        Phi_B = double((rand(k, measure.block_width*measure.block_height)<p));
        Phi_B = Phi_B*2-1;
        Phi_B = Phi_B./(sqrt(measure.block_width*measure.block_height));
        
        Phi=Phi_B;
        Phi_mt = Phi_B*(measure.block_width*measure.block_height/k);
        
    otherwise
        error('Unrecognized measurement model')
        
end

    measure.Phi = Phi;
    measure.Phi_mt = Phi_mt;
    A=@(z)A_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi);
    At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi_mt);

end
