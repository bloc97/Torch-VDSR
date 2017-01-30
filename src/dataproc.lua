require 'image'

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



local dataproc = {}

function dataproc.getImages(num)

	local hr = {}
	local lr = {}

	local imagesn = num or 1 --Number of images in the folder ./train/

	for k=1, imagesn do --Image processing and data augmentation
		
		
		local HRImg = image.rgb2y(image.load("train/" .. k .. ".png", 3, "float"))
		
		local nchan, height, width = HRImg:size(1), HRImg:size(2), HRImg:size(3)
		
		local LRImg = image.scale(image.scale(HRImg, "*1/2"), width, height, "bicubic")

		hr[k] = HRImg
		hr[k+imagesn] = image.hflip(hr[k])
		hr[k+2*imagesn] = image.vflip(hr[k])
		hr[k+3*imagesn] = image.rotate(hr[k], 90)
		
		lr[k] = LRImg
		lr[k+imagesn] = image.hflip(lr[k])
		lr[k+2*imagesn] = image.vflip(lr[k])
		lr[k+3*imagesn] = image.rotate(lr[k], 90)

	end
	
	return hr, lr

end

function dataproc.randCrop(img, img2, w, h)
	local nchan, height, width = img:size(1), img:size(2), img:size(3)
	local cwidth = w or 100
	local cheight = h or 100
	
	local x = math.random(0, width-cwidth)
	local y = math.random(0, height-cheight)
	
	local hr = image.crop(img, x, y, x+cwidth, y+cheight)
	local sr = image.crop(img2, x, y, x+cwidth, y+cheight)
	local s = image.scale(hr, "*1/2")
	
	return hr, sr, s
end

function dataproc.getBatch(hr, lr, n, w, h)
	local cwidth = w or 100
	local cheight = h or 100
	local batchn = n or 10
	
	local tsize = table.getn(hr)
	
	shuffle(hr, lr)
	
	bhr = {}
	blr = {}
	
	bs = {}
	
	for i=1, batchn do
		local k = ((i-1)%tsize)+1
		bhr[i], blr[i], bs[i] = dataproc.randCrop(hr[k], lr[k], cwidth, cheight)
		
	end
	
	return bhr, blr, bs
	
end

return dataproc
























