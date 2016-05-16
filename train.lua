require 'optim'
require 'nn'
require 'rnn'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'cutorch'
require 'paths'

local opts = paths.dofile('opts.lua')
opt = opts.parse(args)
paths.dofile('model.lua')
paths.dofile('dataset.lua')

local model = createModel(opt.nGPU)
local criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()

local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     1,   1e-2,   5e-4, },
        { 2,     4,   5e-3,   5e-4  },
        { 5,     8,   1e-3,   0 },
        { 9,     12,   5e-4,   0 },
        { 12,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
local batchNumber
local loss_epoch

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
     local data, labels = sample(opt.batchSize)
     trainBatch(data, labels)
   end

   cutorch.synchronize()

--   print('==> Evaluating')
--   model:evaluate()

--   local valData, valLabels = read_val()

--   local evalInputs = torch.CudaTensor()
--   local evalLabels = torch.CudaTensor()
--   evalInputs:resize(valData:size()):copy(valData)
--   evalLabels:resize(valLabels:size()):copy(valLabels)

--   local outputs = model:forward(evalInputs)
--   local err = criterion:forward(outputs, evalLabels)

   -- accuracy
--   local correct = 0
--   for i = 1,testData:size()[0] do
--     if outputs[i][valLabels[i]+1] > 0.5 then
--       correct = correct + 1
--     end
--   end

--   local accuracy = correct * 100 / valData:size()[0]

--   print(string.format('Epoch: [%d][VALIDATION] Total Time(s): %.2f\t'
--                          .. 'average loss (per batch): %.2f \t final loss: %.2f'
--                          .. 'accuracy(%%):\t',
--                       epoch, tm:time().real, loss_epoch, err, accuracy))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(1)
   newDPT:add(model:get(1), 1)
   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), newDPT) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local outputs = torch.CudaTensor()

   local err
   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end

   optim.sgd(feval, parameters, optimState)

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end


   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err

  print(outputs)

   -- accuracy
   local correct = 0
   local predictions = outputs:float()
print(predictions)   
for i = 1,opt.batchSize do
     if outputs[i][labelsCPU[i]+1] > 0.5 then
       correct = correct + 1
     end
   end

   local accuracy = correct * 100 / opt.batchSize

   -- Print status
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Acc: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, accuracy,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end


epoch = opt.epochNumber

for i=1,opt.nEpochs do
   train()
   epoch = epoch + 1
end
