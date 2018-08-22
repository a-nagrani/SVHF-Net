function setup_SVHFNet()
%SETUP_SVHFNET Sets up SVHFNet, by adding its folders
% to the Matlab path
%
% Copyright (C) 2018

  vl_setupnn ;
  root = fileparts(mfilename('fullpath')) ;
  addpath(root) ;
  addpath(genpath([root '/mfcc'])) ;
