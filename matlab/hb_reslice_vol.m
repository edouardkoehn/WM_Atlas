function f_o = hb_nii_reslice(f_i,f_r,interp,f_o,SilentMode,FilesInReadOnlyDir)
%HB_NII_RESLICE Coregisteres an input volume to a given reference volume,
% and then reslices the volume so that the volume dimentions match,
% resulting in a one-to-one correspondence between the voxels of the output
% volume and the reference volume. The new resampled volume is written to
% the directory of the input volume, unless name of output file specified.
%
% Inputs:
%   f_i: file to reslice; full path.
%
%   f_r: file to use as ref for resolution & coregistration; full path.
%
%   interp: interpolation order; see spm_reslice.m for details.
%
%   f_o: (optional) name of output file name to save; full path.
%
%   SilentMode: (optional) skip outputing info & also block SPM banners.
%
%   FilesInReadOnlyDir: (optional) 1x2 logical array, specifying wherther
%   f_i and/or of f_r are in read-only directories, respectively. Default:
%   [false false].
%
% Outputs:
%   f_o: resliced file; full path.
%
% Example usage:
%   h_o = hb_nii_reslice(f_i,f_r);
%   hb_nii_reslice(f_i,f_r,1,f_o);
%   hb_nii_reslice(f_i,f_r,1,[],1);
%   hb_nii_reslice(f_i,f_r,1,f_o,[],[0 1]);
%
% Example scenarios:
%
% [Exp-1] if both input files (f_i and f_r) are in writable directory:
%
%         hb_nii_reslice(f_i, f_r, [], f_o);
%
% [Exp-2] if f_i in writable directory but f_r in read-only directory:
%
%         hb_nii_reslice(f_i, f_r, [], f_o, [], [0 1]);
%
% [Exp-3] if both f_i and f_r in read-only directory:
%
%         hb_nii_reslice(f_i, f_r, [], f_o, [], [1 1]);
%
% [Exp-4] auto-naming of output file (only if f_i in writable directory):
%
%         f_o = hb_nii_reslice(f_i, f_r, [], [], [], [0 1]);
%
% Cautionary note on Example [Exp-4]: the auto-naming of output is based on
% two things: i) name of f_i, ii) the voxel resolution of f_r. Therefore,
% for an input f_i, if running the function more than once for two or more
% different f_r that have same voxel resolution (no matter if their names
% differ), output files will be overwritten because the auto-generated
% names will be the same.
%
% Dependencies:
%   SPM12: https://www.fil.ion.ucl.ac.uk/spm/software/spm12
%   spm_run_coreg_hb.m [*]
%   spm_reslice_hb.m [*]
%   [*] https://github.com/aitchbi/matlab-utils/tree/main/spm_modified
%
% Hamid Behjat

if ~exist('interp','var') || isempty(interp)
    interp = 1;
else
    assert(ismember(interp, [0 1]),...
        'Supported interpolation orders: 0 or 1.');
end

if ~exist('f_o','var')
    f_o = [];
end

if ~exist('SilentMode','var') || isempty(SilentMode)
    SilentMode = true;
end

if ~exist('FilesInReadOnlyDir', 'var') || isempty(FilesInReadOnlyDir)
    FilesInReadOnlyDir = [false false]; % [f_i f_r]
end

if ~SilentMode
    fprintf('\n Interpolation order used for reslicing: %d \n',interp);
end

if isstruct(f_i)
    f_i = f_i.fname;
end

%-Define output file name if not given.
%--------------------------------------------------------------------------
if isempty(f_o)

    msg = 'Output file name (full path) needs to be specified.';

    assert(FilesInReadOnlyDir(1)==0, msg);

    h_r = spm_vol(f_r);

    d = round(abs(diag(h_r(1).mat))*1e3);

    if isequal(d(1), d(2), d(3))
        tag = sprintf('%04d', d(1));
    else
        tag = sprintf('%04d_%04d_%04d', d(1), d(2), d(3));
    end

    if endsWith(f_i, '.gz')
        d = strrep(f_i, '.gz', '');
        GzipInput = true;
    else
        d = f_i;
        GzipInput = false;
    end

    [p_i, n_i] = fileparts(d);

    n_o = [n_i, '.res', tag];

    if GzipInput
        f_o = fullfile(p_i, [n_o, '.nii.gz']);
    else
        f_o = fullfile(p_i, [n_o, '.nii']);
    end

    diff_io = false;
else
    if isequal(fileparts(f_i), fileparts(f_o))
        diff_io = false;
    else
        diff_io = true;
    end
end

if endsWith(f_o, '.nii')

    GzipOutput = false;

elseif endsWith(f_o, '.gz')

    f_o = strrep(f_o, '.gz', '');

    GzipOutput = true;
end

%-Use temporary working directory & handle gzips.
%--------------------------------------------------------------------------
if any(FilesInReadOnlyDir)

    TWD = fullfile(...
        fileparts(f_o),...
        sprintf('hb_nii_displace_%s', get_randtag));

    [~,~] = mkdir(TWD);

    if FilesInReadOnlyDir(1)
        f_i = cp2twd(f_i, TWD);
    end

    if FilesInReadOnlyDir(2)
        f_r = cp2twd(f_r, TWD);
    end
else
    TWD = [];
end


[f_r, ~, cleanup_r] = handlegzip(f_r);

[f_i, ~, cleanup_i] = handlegzip(f_i);

%-Define & run job.
%--------------------------------------------------------------------------
job.ref = {strcat(f_r,',1')};

if length(spm_vol(f_i))==1
    job.source = {strcat(f_i,',1')};
else
    job.source = {f_i};
end

job.roptions.interp = interp;
job.roptions.wrap = [0 0 0];
job.roptions.mask = 0;
job.roptions.prefix = 'tmp_';

if diff_io
    job.roptions.writedirectory = fileparts(f_o);
end

if SilentMode
    out = spm_run_coreg_hb(job, true);
else
    out = spm_run_coreg_hb(job);
end

f_tmp = strsplit(out.rfiles{1}, ',');

movefile(f_tmp{1}, f_o); % rename tmp_*.nii to name{f_o}

%-Cleanups.
%--------------------------------------------------------------------------
if cleanup_i
    delete(f_i);
end

if cleanup_r
    delete(f_r);
end

if GzipOutput
    gzip(f_o);
    delete(f_o);
    f_o = [f_o, '.gz'];
end

if ~isempty(TWD)
    rmdir(TWD,'s');
end
end

%==========================================================================
function [f, f_gz, cleanup] = handlegzip(f)
if endsWith(f,'.gz')
    f_gz = f;
    f = strrep(f,'.gz','');
else
    f_gz = [f,'.gz'];
end
if exist(f,'file')
    f_gz = [];
    cleanup = false;
else
    assert(...
        logical(exist(f_gz,'file')),...
        sprintf('GZIP file does not exist: %s',f_gz));
    gunzip(f_gz);
    cleanup = true;
end
end

%==========================================================================
function f2 = cp2twd(f1,TWD)
% copy file to TWD if in read-only dir.
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
function t = get_randtag
t = sprintf('tmp%d',round(rand*1e12));
end
