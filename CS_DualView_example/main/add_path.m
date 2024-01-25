function add_path()

addpath('../Enc');
addpath('../Utilities');
addpath('../Utilities/Measurements');
% 

warning on

addpath('../../../Matlab_Tools/matconvnet-1.0-beta25/matlab');
addpath('../../../Matlab_Tools/matconvnet-1.0-beta25/matlab/simplenn');
addpath('../../../Matlab_Tools/matconvnet-1.0-beta25/matlab/mex');

addpath(genpath('../../Trained_Weights'));
run('../../../Matlab_Tools/matconvnet-1.0-beta25/matlab/vl_setupnn.m');

%% python-env
addpath('..\..\Trained_Weights\Restormer\');

if count(py.sys.path,'W:\Code\WT_code\CS_matlab\Trained_Weights\Restormer\') == 0
    insert(py.sys.path,int32(0),'W:\Code\WT_code\CS_matlab\Trained_Weights\Restormer\');
end

%%
py.importlib.reload(py.importlib.import_module('Restormer_denoise_matlab'));

end