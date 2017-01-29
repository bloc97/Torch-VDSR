require 'torch'
require 'nn'
require 'image'
require 'cltorch'
require 'clnn'

dtype = torch.FloatTensor():type()

vdsrcnn = torch.load("save/nn200.cv")
vdsrcnn:type(dtype)

local trainImg = image.load("test.png", 3, "float")

local trainLRY = image.rgb2y(trainImg):type(dtype)
local trainLR = image.rgb2yuv(trainImg):type(dtype)
local trainDiff = vdsrcnn:forward(trainLRY)

local trainSR = trainLR

trainSR[1]:add(trainDiff[1])

image.save("SR.png", image.yuv2rgb(trainSR))

