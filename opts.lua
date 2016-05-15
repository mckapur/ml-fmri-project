local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('GSF 2016')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-checkpoints',     './checkpoints/', 'directory to write checkpoints and log files into')
    cmd:option('-data',            './data/dataset.hdf5', 'file with dataset')
    cmd:option('-manualSeed',      2, 'Manually set RNG seed')
    cmd:option('-nGPU',            1, 'Number of GPUs to use by default (0 to not use GPU)')

    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       500, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       32,   'mini-batch size (1 = pure stochastic)')

    cmd:option('-LR',              0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')

    cmd:option('-optimState',      'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
end

return M
