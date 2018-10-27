function varargout = maxprojmask(a, b, c, d)
% returns masks for a, b, c, d (or however many inputs u give)
% where mask(a) = 1 where a > b,c,d; 0 else
% usage: [am, bm, cm] = maxprojmask(a, b, c)
varargout = cell(1,nargin);
   
if nargin ==2
    max_ = max(a, b); 
    % Q: threshold needed?!?!
    % level = multithresh(max_, 1); max_(max_<=level) = 0; 
    am = zeros(size(a)); am(a==max_) = 1; varargout{1} = am;
    bm = zeros(size(b)); bm(b==max_) = 1; varargout{2} = bm; 
elseif nargin == 3
    max_ = max(max(a, b), c); 
    am = zeros(size(a)); am(a==max_) = 1; varargout{1} = am;
    bm = zeros(size(b)); bm(b==max_) = 1; varargout{2} = bm; 
    cm = zeros(size(c)); cm(c==max_) = 1; varargout{3} = cm;
elseif nargin == 4
    max_ = max(max(max(a, b),c), d);
    am = zeros(size(a)); am(a==max_) = 1; varargout{1} = am;
    bm = zeros(size(b)); bm(b==max_) = 1; varargout{2} = bm; 
    cm = zeros(size(c)); cm(c==max_) = 1; varargout{3} = cm;
    dm = zeros(size(d)); dm(d==max_) = 1; varargout{4} = dm;
end