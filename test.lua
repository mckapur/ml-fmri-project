paths.dofile('model.lua')
paths.dofile('dataset.lua')

local model = createModel(opt.nGPU)
local criterion = nn.CrossEntropyCriterion()

cutorch.synchronize()

print('==> Testing')
model:evaluate()

local tData, tLabels = read_test()

local testInputs = torch.CudaTensor()
local testLabels = torch.CudaTensor()
testInputs:resize(tData:size()):copy(tData)
testLabels:resize(tLabels:size()):copy(tLabels)

local outputs = model:forward(testInputs)
local err = criterion:forward(outputs, testLabels)

-- accuracy
local correct = 0
for i = 1,tData.size()[0] do
  if outputs[i][tLabels[i]] > 0.5 then
    correct = correct + 1
  end
end

local accuracy = correct * 100 / valData:size()[0]

print(string.format('Final loss: %.2f \t accuracy(%%):\t', err, accuracy))
