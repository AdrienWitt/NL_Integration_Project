function out = print_group_conn(runs, varargin)
%PRINT_GROUP_CONN  Render export_group_conn.py's .surf.nii files with CONN.
%
% Thin wrapper around conn_print_maps.m (writing_cowork, neural_integration
% Study_3), which drives CONN's own conn_mesh_display printer. The curvature
% check is printed first in every run: if it is not a folded brain with light
% gyri and dark sulci, stop -- every other figure in the folder is wrong too.
%
%   print_group_conn                                   % every run with export/conn
%   print_group_conn('k256L8__base_emotionL11__conditional')
%   print_group_conn({'avd__conditional','avd__marginal'}, 'Layout','mosaic8')
%
% Extra Name,Value pairs go to conn_print_maps. Figures land in
% results/group/<run>/export/conn/figs/.

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', 'results', 'group');
printer = fullfile(here, '..', '..', '..', 'writing_cowork', 'projects', ...
    'neural_integration', 'behavioral_scripts_and_data', 'Study_3');
if exist('conn_print_maps', 'file') ~= 2, addpath(printer); end
if exist('conn_print_maps', 'file') ~= 2
    error('conn_print_maps.m not found (looked in %s)', printer);
end

% Headless guard, as in s5_print_contrasts.m: conn_mesh_display reads
% CONN_gui fields that only the CONN GUI populates.
global CONN_gui
if isempty(CONN_gui) || ~isstruct(CONN_gui), CONN_gui = struct(); end
if ~isfield(CONN_gui,'usehighres'),  CONN_gui.usehighres  = false; end
if ~isfield(CONN_gui,'font_offset'), CONN_gui.font_offset = 0;     end
if ~isfield(CONN_gui,'slice_display_skipbookmarkicons')
    CONN_gui.slice_display_skipbookmarkicons = true;
end

if nargin < 1 || isempty(runs)
    d = dir(fullfile(root, '*', 'export', 'conn'));
    runs = unique({d.folder});
    runs = cellfun(@(f) fileparts(fileparts(f)), runs, 'uni', 0);
    [~, runs] = cellfun(@fileparts, runs, 'uni', 0);
end
runs = cellstr(runs);

out = {};
for i = 1:numel(runs)
    cdir = fullfile(root, runs{i}, 'export', 'conn');
    files = dir(fullfile(cdir, '*.surf.nii'));
    if isempty(files), fprintf('[%s] no .surf.nii\n', runs{i}); continue; end
    files = fullfile(cdir, {files.name})';
    % curvature first; sort puts '_check' ahead of letters anyway, be explicit
    iscurv = contains(files, '_check_curvature');
    files = [files(iscurv); files(~iscurv)];
    for k = 1:numel(files)
        if ~conn_surf_dimscheck(files{k})
            error(['%s is not a CONN surface file (dim must be [42 83 94]); ' ...
                   're-run export_group_conn.py'], files{k});
        end
    end
    fprintf('\n[%s] %d maps\n', runs{i}, numel(files));
    out = [out; reshape(conn_print_maps(files, ...
        'OutDir', fullfile(cdir, 'figs'), varargin{:}), [], 1)]; %#ok<AGROW>
end
end
