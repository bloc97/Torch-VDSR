require 'torch'
require 'nn'
require 'image'
require 'cltorch'
require 'clnn'

dtype = torch.FloatTensor():type()

local useOpenCl = false;

if (useOpenCl) then
	require 'cltorch'
	require 'clnn'
	dtype = torch.FloatTensor():cl():type()
end

local saveIters = arg[1]
local name = arg[2]

vdsrcnn = torch.load("save/nn" .. saveIters .. ".cv")
vdsrcnn:type(dtype)


local smallImg = image.load(name .. ".png", 3, "float")

local trainImg = image.scale(smallImg, "*2", "bicubic")

local trainLRY = image.rgb2y(trainImg):type(dtype)
local trainLR = image.rgb2yuv(trainImg):type(dtype)
local trainDiff = vdsrcnn:forward(trainLRY)

local trainSR = trainLR

trainSR[1]:add(trainDiff[1])

image.save(name .. "x2.png", image.yuv2rgb(trainSR))

