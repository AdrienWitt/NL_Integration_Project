function out = print_group_conn(runs, varargin)
%PRINT_GROUP_CONN  Render export_group_conn.py's .surf.nii files with CONN.
%
% Thin wrapper around conn_print_maps.m (writing_cowork, neural_integration
% Study_3), which drives CONN's own conn_mesh_display printer. The curvature
% check is printed first in every run: if it is not a folded brain, stop --
% every other figure in the folder is wrong too.
%
%   print_group_conn                                   % every run with export/conn
%   print_group_conn('k256L8__base_emotionL11__conditional')
%   print_group_conn({'avd__conditional','avd__marginal'}, 'Layout','mosaic8')
%
% Figures land in results/group/<run>/export/conn/fig_<layout>/. Non-negative
% maps get a colorbar starting at 0 (as s5_print_contrasts does); signed maps
% (preference, the curvature check) keep CONN's automatic range. Any other
% Name,Value pair goes to conn_print_maps. Normally driven from Python by
% conn_figures.py, which also handles the MATLAB launch.

p = inputParser; p.KeepUnmatched = true;
p.addParameter('Layout', 'mosaic');
p.addParameter('Stats', {});          % restrict to these statistic names
p.addParameter('Tests', {});          % restrict to these test suffixes
p.addParameter('Vhi', []);            % magnitude maps: fixed bar [0 Vhi]
p.parse(varargin{:});
layout = p.Results.Layout;
stats = cellstr(p.Results.Stats);
tests = cellstr(p.Results.Tests);
vhi = p.Results.Vhi;
extra = namedargs(p.Unmatched);

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
    names = {files.name};
    iscurv = startsWith(names, '_check_curvature');
    if ~isempty(stats) && ~isempty(stats{1})
        keep = iscurv;
        for s = 1:numel(stats), keep = keep | startsWith(names, [stats{s} '__']); end
        names = names(keep); iscurv = iscurv(keep);
    end
    if ~isempty(tests) && ~isempty(tests{1})
        keep = iscurv;
        for s = 1:numel(tests), keep = keep | endsWith(names, ['__' tests{s} '.surf.nii']); end
        names = names(keep); iscurv = iscurv(keep);
    end
    names = [names(iscurv), names(~iscurv)];       % curvature first
    odir = fullfile(cdir, ['fig_' layout]);
    fprintf('\n[%s] %d maps -> %s\n', runs{i}, numel(names), odir);
    for k = 1:numel(names)
        f = fullfile(cdir, names{k});
        if ~conn_surf_dimscheck(f)
            error(['%s is not a CONN surface file (dim must be [42 83 94]); ' ...
                   're-run export_group_conn.py'], f);
        end
        if startsWith(names{k}, '_check_curvature')
            style = {};                                  % CONN's own auto range
        elseif startsWith(names{k}, {'preference', 'valence_vs_arousal'})  % = SIGNED in conn_figures.py
            % Signed: diverging map, symmetric about zero, so neither band is
            % made to look stronger by an asymmetric scale. conn_figures.py
            % redraws this exact colormap for the publication colorbar.
            Y = spm_read_vols(spm_vol(f)); Y = Y(isfinite(Y) & Y ~= 0);
            m = max(abs(Y)); if isempty(m) || m == 0, m = 1; end
            % A shared Vhi gives compared signed maps one [-Vhi Vhi] scale.
            if ~isempty(vhi), m = vhi; end
            style = {'Colormap', 'bluewhitered', 'Vrange', [-m m]};
            % A signed map with its own <stat>.cmap.txt (e.g. a winner map,
            % where the hue must stay visible at small |value|) uses that
            % instead of bluewhitered; still symmetric about zero.
            stat = regexprep(names{k}, '__.*$', '');
            cm = fullfile(cdir, [stat '.cmap.txt']);
            if exist(cm, 'file') == 2
                style = {'Colormap', load(cm, '-ascii'), 'Vrange', [-m m]};
            end
        else
            style = {'Vlo', 0};                          % magnitude: bar from 0
            % A shared [0 Vhi] puts panels that are compared on one scale.
            if ~isempty(vhi), style = {'Vrange', [0 vhi]}; end
            % conn_figures.py writes <stat>.cmap.txt (96x3) for statistics
            % with their own sequential hue; it draws the SVG bar from the
            % same file, so surface and colorbar cannot disagree.
            stat = regexprep(names{k}, '__.*$', '');
            cm = fullfile(cdir, [stat '.cmap.txt']);
            if exist(cm, 'file') == 2
                style = [style, {'Colormap', load(cm, '-ascii')}]; %#ok<AGROW>
            end
        end
        w = conn_print_maps({f}, 'OutDir', odir, 'Layout', layout, style{:}, extra{:});
        out = [out; reshape(w, [], 1)]; %#ok<AGROW>
    end
end
end

function c = namedargs(s)
f = fieldnames(s); c = cell(1, 2*numel(f));
for i = 1:numel(f), c{2*i-1} = f{i}; c{2*i} = s.(f{i}); end
end
