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

local inImg = image.scale(smallImg, "*2", "bicubic")

local inDiff = vdsrcnn:forward(inImg:clone():add(-0.5))

local outImg = inImg

outImg:add(inDiff)

image.save(name .. "x2.png", outImg)

