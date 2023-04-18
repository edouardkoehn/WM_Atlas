function out = spm_run_coreg_hb(job,silentMode)
% SPM job execution function
% takes a harvested job data structure and call SPM functions to perform
% computations on the data.
% Input:
% job    - harvested job data structure (see matlabbatch help)
% Output:
% out    - computation results, usually a struct variable.
%__________________________________________________________________________
% Copyright (C) 2005-2014 Wellcome Trust Centre for Neuroimaging

% $Id: spm_run_coreg.m 5956 2014-04-16 14:34:25Z guillaume $







if ~exist('silentMode','var') || isempty(silentMode)
    silentMode = false;
end

% HB:
% [July 2017] added option to write resulting file in a different locations than
% default.
% [Aug 2021] added option for silentMode rnning (skipping SPM banners).
%
% Hamid Behjat.





if ~isfield(job,'other') || isempty(job.other{1}), job.other = {}; end
PO = [job.source(:); job.other(:)];
PO = spm_select('expand',PO);

%-Coregister
%--------------------------------------------------------------------------
if isfield(job,'eoptions')
    x  = spm_coreg(char(job.ref), char(job.source), job.eoptions);

    M  = spm_matrix(x);
    MM = zeros(4,4,numel(PO));
    for j=1:numel(PO)
        MM(:,:,j) = spm_get_space(PO{j});
    end
    for j=1:numel(PO)
        spm_get_space(PO{j}, M\MM(:,:,j));
    end
end

%-Reslice
%--------------------------------------------------------------------------
if isfield(job,'roptions')
    P            = char(job.ref{:},job.source{:},job.other{:});
    flags.mask   = job.roptions.mask;
    flags.mean   = 0;
    flags.interp = job.roptions.interp;
    flags.which  = 1;
    flags.wrap   = job.roptions.wrap;
    flags.prefix = job.roptions.prefix;

    %HB--------------------------------------------------------------------
    if isfield(job.roptions,'writedirectory')
        flags.saveaddress = job.roptions.writedirectory;
    end
    if silentMode
        spm_reslice_hb(P,flags,true);
    else
        spm_reslice_hb(P,flags);
    end
    %--------------------------------------------------------------------HB
end

%-Dependencies
%--------------------------------------------------------------------------
if isfield(job,'eoptions')
    out.cfiles   = PO;
    out.M        = M;
end
if isfield(job,'roptions')
    out.rfiles   = spm_file(PO, 'prefix',job.roptions.prefix);
end
