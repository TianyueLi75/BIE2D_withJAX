function s = mergesegquads(sarray)
% MERGESEGQUADS  combine the quadrature info from segments into one (not Z,Zp..)
%
% s = mergesegquads(sarray) where sarray is a struct array of segments,
%  outputs s a single struct containing the concatenated quadrature point info.
%  The fields t, Z, Zp, etc are not handled.

% Barnett 12/17/17
% Choco Feb 2026: include t as merged fields for close eval.
s = sarray(1); s = rmfield(s,{'Z','Zp','Zpp'});
for i=2:numel(sarray)
  t = sarray(i);
  s.x = [s.x;t.x]; s.nx = [s.nx;t.nx]; s.xp = [s.xp;t.xp];
  s.t = [s.t;t.t];
  s.xpp = [s.xpp;t.xpp]; s.sp = [s.sp;t.sp]; s.tang = [s.tang;t.tang];
  s.cur = [s.cur;t.cur]; s.w = [s.w;t.w]; s.cw = [s.cw;t.cw];
end
