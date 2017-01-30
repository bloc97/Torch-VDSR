require 'torch'
require 'nn'
local nninit = require 'nninit'

--Private Methods
local getBias = function(nnl)
  return nnl.bias
end

local InConvolution = function()
	return nn.SpatialConvolutionMM(1, 64, 3, 3, 1, 1, 1, 1)
end

local HiddenConvolution = function()
	return nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1)
end

local OutConvolution = function()
	return nn.SpatialFullConvolution(64, 1, 3, 3, 1, 1, 1, 1)
end

local nonLinear = function()
	return nn.ELU(0.1, false) --ELU has better convergence
	--return nn.ReLU(true)
end

--Module start

local model = {}

function model.create(depth)

	--Start of model
	local vdsrcnn = nn.Sequential()

	vdsrcnn:add(nn.SpatialConvolutionMM(1, 56, 5, 5, 1, 1, 2, 2))
	vdsrcnn:add(nonLinear())
	vdsrcnn:add(nn.SpatialConvolutionMM(56, 12, 1, 1, 1, 1, 0, 0))
	vdsrcnn:add(nonLinear())
	vdsrcnn:add(nn.SpatialConvolutionMM(12, 12, 3, 3, 1, 1, 1, 1))
	vdsrcnn:add(nonLinear())
	vdsrcnn:add(nn.SpatialConvolutionMM(12, 12, 3, 3, 1, 1, 1, 1))
	vdsrcnn:add(nonLinear())
	vdsrcnn:add(nn.SpatialConvolutionMM(12, 12, 3, 3, 1, 1, 1, 1))
	vdsrcnn:add(nonLinear())
	vdsrcnn:add(nn.SpatialConvolutionMM(12, 12, 3, 3, 1, 1, 1, 1))
	vdsrcnn:add(nonLinear())
	vdsrcnn:add(nn.SpatialConvolutionMM(12, 56, 1, 1, 1, 1, 0, 0))
	vdsrcnn:add(nonLinear())
	vdsrcnn:add(nn.SpatialFullConvolution(56, 1, 9, 9, 2, 2, 4, 4, 1, 1))


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