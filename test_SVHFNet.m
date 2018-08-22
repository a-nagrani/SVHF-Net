function test_SVHFNet(varargin)
% minimal test demo with the SVHFNet model pretrained on
% the VoxCeleb dataset for binary cross-modal matching


% ----------------------------------------------------
% download model
% ----------------------------------------------------
opts.modelPath = '';
modelName = 'static-RGB+VGGFace.mat' ;
paths = {opts.modelPath, ...
    modelName, ...
    fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

if isempty(ok)
    fprintf('Downloading SVHFNet for binary face-voice matching ... this may take a while\n') ;
    opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
    mkdir(fileparts(opts.modelPath)) ; base = 'http://www.robots.ox.ac.uk' ;
    url = sprintf('%s/~vgg/research/CMBiometrics/models/%s', base, modelName) ;
    urlwrite(url, opts.modelPath) ;
else
    opts.modelPath = paths{ok} ;
end


netStruct = load(opts.modelPath);
net = dagnn.DagNN.loadobj(netStruct.net);
net.mode = 'test';

% ----------------------------------------------------
% settings
% ----------------------------------------------------
opts.gpu = 1;
opts.numThreads = 4;
opts.imageSize = net.meta.face.normalization.imageSize;
opts.subtractAverage = net.meta.face.normalization.averageImage;
opts.cropSize = 0.85;
opts.dataDir = 'files';

buckets.pool     = [2 5 8 11 14 17 20 23 27 30];
buckets.width     = [100 200 300 400 500 600 700 800 900 1000];

if ~isempty(opts.gpu), net.move('gpu'); end
net.conserveMemory = false;


% ----------------------------------------------------
% read test files and evaluate
% ----------------------------------------------------
audiopath = fullfile(opts.dataDir, 'audio1.wav');
facepath1 = fullfile(opts.dataDir, 'face1.jpg');
facepath2 = fullfile(opts.dataDir, 'face2.jpg');

inp_a = test_getinput_audio({audiopath},net.meta.voice, buckets);
inp_f1 = test_getinput_face({facepath1}, opts);
inp_f2 = test_getinput_face({facepath2}, opts);

%x contains softmax predictions for both faces
x = evaluate_net_avgpool(net, buckets, gpuArray(inp_f1{1}),gpuArray(inp_f2{1}),gpuArray(inp_a{1})) ;
[score, class] = max(x) ;
fprintf('prediction for triplet | class %d, confidence: %g\n', class, score) ;

function inp = test_getinput_face(images,opts)
args{1} = {images, ...
    'NumThreads', opts.numThreads, ...
    'Pack', ...
    'Interpolation', 'bicubic', ...
    'Resize', opts.imageSize(1:2) ...
    'CropSize', opts.cropSize
                        } ;
    args{end+1} = {'Gpu'} ;
    args{end+1} = {'SubtractAverage', opts.subtractAverage} ;
    args = horzcat(args{:}) ;
    inp = vl_imreadjpeg(args{:}) ;

 end

function inp = test_getinput_audio(images,meta,buckets)
for i = 1: numel(images)
    audfile = images{i};

    z             = audioread(audfile);

    SPEC         = runSpec(z,meta.audio);
    mu            = mean(SPEC,2);
    stdev         = std(SPEC,[],2) ;
    nSPEC         = bsxfun(@minus, SPEC, mu);
    nSPEC         = bsxfun(@rdivide, nSPEC, stdev);

    rsize     = buckets.width(find(buckets.width(:)<=size(nSPEC,2),1,'last'));
    rstart  = round((size(nSPEC,2)-rsize)/2);

    inp{i} = single(nSPEC(:,rstart:rstart+rsize-1));

end
end


function x = evaluate_net_avgpool(net,buckets,image_f1,image_f2,image_v)
s1 = size(image_v,2);
p1 = buckets.pool(s1==buckets.width);
ind1 = strcmp({net.layers(:).name}, 'pool6_voice');
net.layers(ind1).block.poolSize=[1 p1];  % change the average pool layer size depending on the length of the test audio segment

net.vars(net.getVarIndex('softmax')).precious = true;
net.eval({'input_face1',image_f1, 'input_face2', image_f2,'input_voice', image_v});

x = gather(net.vars(net.getVarIndex('softmax')).value);

end
end
