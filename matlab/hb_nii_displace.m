function hb_nii_displace(f_s,f_d,f_r,f_o,varargin)
% HB_DISPLACE_NII Transforms a nifti volume from one space to another using
% a displacement field.
%
% Inputs:
%   f_s: full path to nifti file to be converted; source file.
%
%   f_d: full path of nifti file containing the displacement field used in
%   the space conversion; 4D volume of length 3. For example, for HCP data,
%   if f_s is in ACPC and f_r is in MNI, f_d is the acpc_dc2standard.nii
%   displacement map.
%
%   f_r: full path to file used as reference for the dimensions
%   of the output.
%
%   f_o: full path of the file that will be created and saved.
%
%   Name-Value Pair Arguments:
%
%   WhichVols: (n/a if input is a single volume) vector of indices of
%   volumes to convert from a 4D f_s.
%
%   Verbose: logical; show progress? (default: true)
%
%   CopyAllFilesToTempDir: logical; (default: false) This options allows
%   working with input files that are located in read-only directores;
%   files will first be duplicated into a temp directory, worked on, and
%   then cleaned up.
%
% Dependencies:
%   spm12, to read/write nifti files.
%   https://www.fil.ion.ucl.ac.uk/spm/software/spm12
%
% David Abramian
% Hamid Behjat

p = inputParser;
addParameter(p,'Verbose',true);
addParameter(p,'WhichVols',[]);
addParameter(p,'InputFilesInReadOnlyDir',false);

parse(p,varargin{:});
opts = p.Results;

% temporary wrute directory
if endsWith(f_o,'.nii')
    TWD = fileparts(f_o);
    GzipOutput = false;
elseif endsWith(f_o,'.gz')
    TWD = fileparts(strrep(f_o,'.gz',''));
    GzipOutput = true;
    f_o = strrep(f_o,'.gz','');
end
TWD = fullfile(TWD,sprintf('hb_nii_displace_%s',get_randtag));
[~,~] = mkdir(TWD);

% copy files to TWD
if opts.InputFilesInReadOnlyDir
    f_s = cp2twd(f_s, TWD);
    f_r = cp2twd(f_r, TWD);
    f_d = cp2twd(f_d, TWD);
end


% prepare source file
f_s = prep_src(f_s,TWD);
% mk output dir
[~,~] = mkdir(fileparts(f_o));
% vols to process
if isempty(opts.WhichVols)
    opts.WhichVols = 1:length(spm_vol(f_s));
end
Nv = length(opts.WhichVols);

%-Displacement file.
f_d = handlegzip(f_d);
h_d = spm_vol(f_d);
v_d = spm_read_vols(h_d);
mat_d = h_d.mat;

%-Reference file.
f_r = handlegzip(f_r);
h_r = spm_vol(f_r);
if length(h_r)>1
    h_r = h_r(1);
end
mat_r = h_r.mat;
matinv_r = abs(inv(mat_r));
matinv_r(1:3,4) = 0;

dim_o = h_r.dim;

res_matinv_r = diag(matinv_r(1:3,1:3))';

for iV = 1:Nv

    % source file
    d = strcat(f_s,',',num2str(opts.WhichVols(iV)));
    h_s = spm_vol(d);
    mat_s = h_s.mat;

    if iV==1
        % voxel coordinates of output file
        [yy_o,xx_o,zz_o] = meshgrid(1:dim_o(2),1:dim_o(1),1:dim_o(3));

        %-Step1: ref space >> disp map
        % positions in disp map
        A = affine3d((mat_d\mat_r)');
        [yy_d,xx_d,zz_d] = transformPointsForward(A,yy_o,xx_o,zz_o);

        % interpolate disp map to find values at positions
        disp_xx = interp3(v_d(:,:,:,1),yy_d,xx_d,zz_d);
        disp_yy = interp3(v_d(:,:,:,2),yy_d,xx_d,zz_d);
        disp_zz = interp3(v_d(:,:,:,3),yy_d,xx_d,zz_d);

        %-Step2: disp map >> ref space
        % positions in ref space
        xx_r = xx_o + res_matinv_r(1)*disp_xx;
        yy_r = yy_o + res_matinv_r(2)*disp_yy;
        zz_r = zz_o + res_matinv_r(3)*disp_zz;

        %-Step3: ref space >> source space
        % positions in source space
        A = affine3d((mat_s\mat_r)');
        [yy_s,xx_s,zz_s] = transformPointsForward(A,yy_r,xx_r,zz_r);

        mat_s_iV1 = mat_s;
    else
        % Step3 assumes header mats are the same across all frames; if
        % not, Step3 should be computed per frame.
        if not(isequal(mat_s,mat_s_iV1))
            A = affine3d((mat_s\mat_r)');
            [yy_s,xx_s,zz_s] = transformPointsForward(A,yy_r,xx_r,zz_r);
        end
    end

    v_s = spm_read_vols(h_s);
    v_s(isnan(v_s)) = 0; % drop NaNs; interp fails due to erosion around NaNs

    % interpolate source file to find values at positions
    v_o = interp3(v_s,yy_s,xx_s,zz_s);
    %v_o(isnan(v_o)) = 0;
    v_o(v_o==0)=NaN;

    %-Write output
    if iV==1
        h_o = struct;
        if not(isempty(TWD))
            [~,n,e] = fileparts(f_o);
            while 1
                f_o_tmp = fullfile(TWD,[n,'_',get_randtag,e]);
                if ~exist(f_o_tmp,'file')
                    break;
                end
            end
            h_o.fname = f_o_tmp;
        else
            h_o.fname = f_o;
        end
        h_o.dim   = h_r.dim;
        h_o.mat   = h_r.mat;
        h_o.dt    = h_s.dt;
        h_o = spm_create_vol(h_o);
    end
    h_o.n(1) = iV;
    if ~isempty(h_s.private.timing)
        h_o.private.timing = h_s.private.timing;
    end
    spm_write_vol(h_o,v_o);

    if opts.Verbose && Nv>1
        showprgs(iV,Nv);
    end
end

% transfer from tmp to dest dir
if not(isempty(TWD))
    movefile(f_o_tmp,f_o);
end

if GzipOutput
    gzip(f_o);
    delete(f_o);
end

% remove tmp dir
rmdir(TWD,'s');
end

%==========================================================================
function [f_tmp] = prep_src(f,TWD)
% Does two things:
% - Copies f to TWD, if f & TWD are not on the same machine.
% - If f is gziped, gunzips it, and drops '.gz' from f.
%
% At least when f exists in gziped form, its faster to copy the gzip to
% local, gunzip on local and proces, than to work on remote. For large 4D
% volumes, reading/writing from/to remote disk while running computations
% on a local server can be very slow, depending on how the remote machine
% (e.g. NAS) is connected to the server.

% verify file
if not(exist(f,'file'))
    if endsWith(f,'.nii')
        % maybe gzip version exists?

        f = strrep(f,'.nii','.nii.gz');

        assert(...
            logical(exist(f,'file')),...
            sprintf('Input file missing: %s',f));

    elseif endsWith(f,'.gz')
        % maybe non-gzip version exists?

        f = strrep(f,'.gz','');

        assert(...
            logical(exist(f,'file')),...
            sprintf('Input file missing: %s',f));

    else
        error('Unrecongnized input file format.');
    end
end

if endsWith(f,'.gz')
    GZIPFILE = true;
else
    GZIPFILE = false;
end

if not(GZIPFILE)
    % check if gziped f exists, if so, copy that instead; much faster.
    d = [f,'.gz'];
    if exist(d,'file')
        f = d;
        GZIPFILE = true;
    end
end

% copy fie to TWD
if isequal(fileparts(f),TWD)
    f_tmp = f;
else
    f_tmp = cp2twd(f,TWD);
end

if GZIPFILE
    gunzip(f_tmp);
    delete(f_tmp); % delete '.gz' version to free up space
    f_tmp = strrep(f_tmp,'.gz','');
end
end

%==========================================================================
function f2 = cp2twd(f1,TWD)
d = strrep(f1, fileparts(f1), TWD);
while 1 % NOTE
    f2 = strrep(d, '.nii', ['_', get_randtag, '.nii']);
    if ~exist(f2,'file')
        break
    end
end
sts = copyfile(f1,f2);
assert(sts==1);
% NOTE: random tag for robustness, e.g. in parallel runs.
end

%==========================================================================
function showprgs(n,N)
l = numel(num2str(N));
if n==1
    fprintf('\n..Mapping volume using displacement map.. ');
else
    fprintf(repmat('\b',1,2*l+1),n);
end
eval(['fprintf(''%-',num2str(l),'d/%-',num2str(l),'d'',n,N)'])
end

%==========================================================================
function f = handlegzip(f)
if endsWith(f,'.gz')
    gunzip(f);
    f = strrep(f,'.gz','');
end
end

%==========================================================================
function t = get_randtag
t = sprintf('tmp%d',round(rand*1e12));
end
