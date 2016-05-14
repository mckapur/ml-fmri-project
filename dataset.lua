require 'hdf5'

-- Open the file, read metadata
local datafile = hdf5.open(opt.data)
local f = datafile:read('/metadata'):all()
local nAutisticTrain = f[1]
local nControlTrain = f[2]
local nAutisticVal = f[3]
local nControlVal = f[4]
local nAutisticTest = f[5]
local nControlTest = f[6]

function sample(n)
  local data = torch.Tensor(n, 1, 60, 60, 60)
  local labels = torch.Tensor(n)
  for i = 1,n do
    if torch.uniform() < 0.5 then
      data[i] = datafile:read('/autistic/train/' + torch.random(1, nAutisticTrain))
      labels[i] = 1
    else
      data[i] = datafile:read('/control/train/' + torch.random(1, nControlTrain))
      labels[i] = 0
    end
  end
  return data, labels
end

function read_val()
  local n = nAutisticVal + nControlVal
  local data = torch.Tensor(n, 1, 60, 60, 60)
  local labels = torch.Tensor(n)
  for i = 1,nAutisticVal do
    data[i] = datafile:read('/autistic/val/' + i)
    labels[i] = 1
  end
  for i =1,nControlVal do
    data[nAutisticVal+i] = datafile:read('/control/val' + i)
    labels[nAutisticVal+i] = 0
  end
  return data, labels
end

function close()
  datafile:close()
end
