require 'torch'
require 'nn'
require 'optim'
require 'image'
--require 'hdf5'
require 'nninit'
--Use FloatTensor for faster training
local dtype = 'torch.FloatTensor'

local useOpenCl = true;

--If we are using opencl, we change the tensor dtype to "ClTensor" using :cl();
if (useOpenCl) then
	require 'cltorch'
	require 'clnn'
	dtype = torch.FloatTensor():cl():type()
end

--Create Loss Function
local criterion = nn.MSECriterion():type(dtype)
criterion.sizeAverage = false

--Create VDSR conv neural network
--http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf

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
	return nn.SpatialConvolutionMM(64, 1, 3, 3, 1, 1, 1, 1)
end

local nonLinear = function()
	return nn.ELU(0.1, false) --ELU has better convergence
	--return nn.ReLU(true)
end

--Start of model
vdsrcnn = nn.Sequential()

vdsrcnn:add(InConvolution())
vdsrcnn:add(nonLinear())
for nnlayers = 1, 8 do
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

--Set the network to the dtype
vdsrcnn:type(dtype)

--Create training data

function TableToTensor(table)
  local tensorSize = table[1]:size()
  local tensorSizeTable = {-1}
  for i=1,tensorSize:size(1) do
    tensorSizeTable[i+1] = tensorSize[i]
  end
  merge=nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(unpack(tensorSizeTable)))

  return merge:forward(table)
end

function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end

function shuffle(array, array2)
    local counter = #array
    while counter > 1 do
        local index = math.random(counter)
        swap(array, index, counter)
        swap(array2, index, counter)
        counter = counter - 1
    end
end

function subrange(t, first, last)
  local subt = {}
  for i=first,last do
    subt[#subt + 1] = t[i]
  end
  return subt
end

local hr = {}
local lr = {}

local imagesn = 80 --Number of images in the folder ./train/


for k=1, imagesn do --Image processing and data augmentation
	local HRImgCropped = image.crop(image.load("train/" .. k .. ".png", 3, "float"),0,0,100,100) --Crop a 100x100 location from the image
	local LRImgCropped = image.scale(image.scale(HRImgCropped, "*1/2"), "*2", "bicubic")

	hr[k] = image.rgb2y(HRImgCropped)
	hr[k+imagesn] = image.hflip(hr[k])
	hr[k+2*imagesn] = image.vflip(hr[k])
	hr[k+3*imagesn] = image.rotate(hr[k], 90)
	
	lr[k] = image.rgb2y(LRImgCropped)
	lr[k+imagesn] = image.hflip(lr[k])
	lr[k+2*imagesn] = image.vflip(lr[k])
	lr[k+3*imagesn] = image.rotate(lr[k], 90)

end

--shuffle(hr, lr) --shuffle the data

local x;
local y;

local batchsize = 10 --Reduce the batch size if you have memory problems (C++ Exception or Out of memory error)
local minibatch = (imagesn*4)/batchsize --#Of iterations before going through entire batch

function setBatch(iter)
	local batch = iter%minibatch
	local start = batch*batchsize+1
	local eloc = batch*batchsize+batchsize
	
	x = TableToTensor(subrange(lr, start, eloc)):type(dtype)
	y = TableToTensor(subrange(hr, start, eloc)):type(dtype)

end

setBatch(0)




--Initialise training variables

params, gradParams = vdsrcnn:getParameters()

local optimState = {learningRate = 0.1, weightDecay = 0.0001, momentum = 0.9}
local cnorm = 0.001 * optimState.learningRate --Gradient Clipping (c * Initial_Learning_Rate)

local showlossevery = 100;
local loss = 1;

--Training function
function f(params)

	--vdsrcnn:zeroGradParameters();
	gradParams:zero()
	
	local imagein = x:clone():csub(0.5) --Removing 0.5 to normalise the input images to [-0.5, 0.5] helps prevent gradient explosion, as if the image has values of [0, 1], all the gradients initially will be positive at the same time
	
	--Forward the image values
	local out = vdsrcnn:forward(imagein)
	local diff = y:clone():csub(x)
	
	--The loss is the difference between the output residual and the ground truth residual
	loss = criterion:forward(out, diff)
	
	--Compute the gradient
	local lrate = optimState.learningRate
	local grad_out = criterion:backward(out, diff)
	
	--Zero the previous gradient, and backpropagate the new gradient
	local grad_in = vdsrcnn:backward(imagein, grad_out):clamp(-cnorm/lrate, cnorm/lrate)
	
	gradParams:clamp(-cnorm/lrate, cnorm/lrate) --Clip the gradients

	--Return the loss and new gradient parameters to the optim.sgd() function
	return loss, gradParams
end

local decreaseRate = 0.1

--Saves a ground truth residual for testing
local Truthdiff = hr[1]:clone():csub(lr[1])
image.save("test/Tresid.png", Truthdiff:add(0.5))


local epoch = 0;

for iter = 1, 30000 do
	if (iter%10000 == 0) then
		optimState.learningRate = optimState.learningRate * decreaseRate
		print("Reducing learning rate by a factor of " .. decreaseRate .. ". New learning rate: " .. optimState.learningRate)
	end
	optim.sgd(f, params, optimState)
	
	if ((iter%showlossevery == 0) or (iter%20 == 0 and iter < 200) or (iter < 20)) then --Print the training loss and an example residual output to compare with ground truth
		print("Epoch " .. epoch .. " Iteration " .. iter .. " Training Loss " .. loss)
		
		local epochdiff = vdsrcnn:forward(lr[1]:type(dtype))
		image.save("test/" .. iter .. "resid.png", epochdiff:add(0.5))
	end
	
	if (iter%100 == 0) then --save model each 100 iterations
		vdsrcnn:clearState()
		vdsrcnn:float()
		torch.save("save/nn" .. iter .. ".cv", vdsrcnn)
		vdsrcnn:type(dtype)
		params, gradParams = vdsrcnn:getParameters()
		collectgarbage()
		
	end
	
	if (iter%minibatch == minibatch-1) then
		epoch = epoch+1
	end
	setBatch(iter)

end




