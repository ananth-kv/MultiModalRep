local utils = require 'misc.utils'
local net_utils = {}

-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)
    if i == layer_num - 3 then --Ananth
      break
    end
    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    --print('layer: ', layer)
    cnn_part:add(layer)
  end

  cnn_part:add(nn.Linear(4096,encoding_size))
  cnn_part:add(backend.ReLU(true))
  --print('layer: ',nn.Linear(4096,encoding_size))
  --print('layer: ',backend.ReLU(true))
  --cnn_part:add(nn.Linear(2400,encoding_size)) --Ananth
  --cnn_part:add(backend.ReLU(true)) --Ananth
  return cnn_part
end


-- Builds Discriminator network
function net_utils.build_netD()
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  
    if backend == 'cudnn' then
       require 'cudnn'
       backend = cudnn
    elseif backend == 'nn' then
      require 'nn'
      backend = nn
    else
      error(string.format('Unrecognized backend "%s"', backend))
    end
    
    local model = nn.Sequential() 
    
    model:add(nn.Linear(2400, 1000))                  -- Fully connected layer, 2400 inputs, 1000 outputs
    model:add(nn.LeakyReLU(0.2, true))
    
    model:add(nn.Linear(1000, 500))
    model:add(nn.LeakyReLU(0.2, true))
    
    model:add(nn.Linear(500, 128))
    model:add(nn.LeakyReLU(0.2, true))

    model:add(nn.Linear(128, 2))                     -- Final layer has 2 outputs. One for image wave, one for text
    model:add(nn.LeakyReLU(0.2, true))
    model:add(nn.LogSoftMax())                      -- log-probability output, since this is a classification problem
    
    return model
end


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

-- builds CNN from scratch -  Note: VGG-16 SPECIFIC!
function net_utils.build_cnn_scratch()
   local modelType = 'D' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            features:add(conv3)
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   features:cuda()

    local decod = nn.Sequential()
    local SpatialBatchNormalization = nn.SpatialBatchNormalization
    local SpatialFullConvolution = nn.SpatialFullConvolution

    local nc = 3
    local ngf = 32
    local nz = 512
    -- creates 224X224 images
    decod:add(SpatialFullConvolution(nz, ngf * 8, 2, 2, 2, 2))
    decod:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true)) --10*256*16*16

    decod:add(SpatialFullConvolution(ngf*8, ngf * 4, 4, 4, 2, 2, 1, 1))
    decod:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true)) --10*128*32*32

    decod:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1)) 
    decod:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true)) --10*64*64*64

    decod:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    decod:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true)) --10*32*112*112

    decod:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1)) --10*3*256*256
    decod:add(nn.Tanh())

    decod:cuda()
    
    -- creates 64X64 images
    local netG = nn.Sequential()
    netG:add(SpatialFullConvolution(nz, ngf * 8, 2, 2))
    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true)) -- 8
-- state size: (ngf*8) x 4 x 4
    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true)) -- 16
-- state size: (ngf*4) x 8 x 8
    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true)) -- 32
-- state size: (ngf*2) x 16 x 16
    netG:add(SpatialFullConvolution(ngf * 2, nc, 4, 4, 2, 2, 1, 1))
    netG:add(nn.Tanh())
    
    netG:cuda()
    
    
    
    local model = nn.Sequential()
    model:add(features):add(decod)
    model.imageSize = 256
    model.imageCrop = 224

    model:apply(weights_init)

   --local classifier = nn.Sequential()
   --classifier:add(nn.View(512*7*7))
   --classifier:add(nn.Linear(512*7*7, 4096))
   --classifier:add(nn.Threshold(0, 1e-6))
   --classifier:add(nn.Dropout(0.5))
   --classifier:add(nn.Linear(4096, 4096))
   --classifier:add(nn.Threshold(0, 1e-6))
   --classifier:add(nn.Dropout(0.5))
   --classifier:add(nn.Linear(4096, nClasses))
   --classifier:add(nn.LogSoftMax())
   --classifier:cuda()

   --local model = nn.Sequential()
   --model:add(features):add(classifier)
   --model.imageSize = 256
   --model.imageCrop = 224

   return model
end


-- Builds fractionally strided convolution network / Decoder for CNN
function net_utils.build_fsc()
    
    local decod = nn.Sequential()
    local SpatialBatchNormalization = nn.SpatialBatchNormalization
    local SpatialFullConvolution = nn.SpatialFullConvolution

    local nc = 3
    local ngf = 32
    local nz = 512
    
    decod:add(SpatialFullConvolution(nz, ngf * 8, 2, 2, 2, 2))
    decod:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true)) --10*256*16*16

    decod:add(SpatialFullConvolution(ngf*8, ngf * 4, 4, 4, 2, 2, 1, 1))
    decod:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true)) --10*128*32*32

    decod:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1)) 
    decod:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true)) --10*64*64*64

    decod:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    decod:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true)) --10*32*112*112

    decod:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1)) --10*3*256*256
    decod:add(nn.Tanh())
    
    return decod
end


-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  end
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

  return imgs
end

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end
function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

return net_utils
