require 'torch'
require 'nn'
local nninit = require 'nninit'

--Private Methods
local getBias = function(nnl)
  return nnl.bias
end

local InConvolution = function()
	return nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 1, 1)
end

local HiddenConvolution = function()
	return nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1)
end

local OutConvolution = function()
	return nn.SpatialConvolutionMM(64, 3, 3, 3, 1, 1, 1, 1)
end

local nonLinear = function()
	return nn.ELU(0.3, false) --ELU has better convergence
	--return nn.ReLU(true)
end

--Module start

local model = {}

function model.create(depth)

	--Start of model
	local vdsrcnn = nn.Sequential()

	vdsrcnn:add(InConvolution())
	vdsrcnn:add(nonLinear())
	for layers = 1, depth do
		vdsrcnn:add(HiddenConvolution())
		vdsrcnn:add(nonLinear()) --ReLU or ELU
	end
	vdsrcnn:add(OutConvolution())


	local function weights_init(m)
	   local name = torch.type(m)
	   if name:find('Convolution') then
		  m:init('weight', nninit.kaiming, {dist = 'normal', gain = {'lrelu', leakiness = 0.1}}) --He et al. Initialisation
		   :init(getBias, nninit.constant, 0)
	   end
	end

	vdsrcnn:apply(weights_init)
	
	return vdsrcnn
end

return model