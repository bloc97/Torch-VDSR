require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nninit'

local model = require 'src/model'
local dataproc = require 'src/dataproc'


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

vdsrcnn = model.create(8)
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

local imagesn = 12 --Number of images in the folder ./train/
local batchsize = 10 --Reduce the batch size if you have memory problems (C++ Exception or Out of memory error)
local minibatch = (imagesn*4)/batchsize --#Of iterations before going through entire batch

local hr, lr = dataproc.getImages(imagesn)
local timg = image.load("train/test.png", 3, "float")
local thr = image.rgb2y(timg):type(dtype)
local tlr = image.rgb2y(image.scale(image.scale(timg, "*1/2"), thr:size(3), thr:size(2), "bicubic")):type(dtype)

local x;
local y;


function setBatch()
	ay, ax = dataproc.getBatch(hr, lr, n, w, h)
	x = TableToTensor(ax):type(dtype)
	y = TableToTensor(ay):type(dtype)
end

setBatch()




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
	
	local imagein = x
	
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
local Truthdiff = thr:clone():csub(tlr)
image.save("test/Truth.png", Truthdiff:add(0.5))


--image.save("test/TestInput.png", x[1])
--image.save("test/TestOutput.png", y[1])

local Truthdiff2 = y[1]:clone():csub(x[1])
image.save("test/TestGT1.png", Truthdiff2:add(0.5))

local epoch = 0;

for iter = 1, 30000 do
	if (iter%10000 == 0) then
		optimState.learningRate = optimState.learningRate * decreaseRate
		print("Reducing learning rate by a factor of " .. decreaseRate .. ". New learning rate: " .. optimState.learningRate)
	end
	optim.sgd(f, params, optimState)
	
	if ((iter%showlossevery == 0) or (iter%20 == 0 and iter < 200) or (iter < 20)) then --Print the training loss and an example residual output to compare with ground truth
		print("Epoch " .. epoch .. " Iteration " .. iter .. " Training Loss " .. loss)
		
		local epochdiff = vdsrcnn:forward(tlr)
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
	setBatch()

end




