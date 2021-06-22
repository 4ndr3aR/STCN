from PIL import Image
import numpy as np

def blend_img_mask(img_fn, mask, save_fn, alpha=0.1): 
	img  = np.array(Image.open(img_fn))
	mask = np.array(mask.convert('RGB'))
	if mask.shape[0] == 1:
		mask = np.stack((mask,) * 3, axis=-1)
	#img = img.convert('RGBA', img)
	#combo = Image.blend(img, mask, alpha=0.5)
	combo = Image.fromarray(np.uint8(img * (1.0 - alpha) + mask * alpha))
	print(f'Saving image: {save_fn}')
	combo.save(save_fn)


if __name__ == "__main__":
	img_fn  = '/home/ranieri/dataset/ericsson-camera-360/ericsson-camera-360-4k-frames-dummy/JPEGImages/video1/ericsson-camera-360-05.jpg'
	mask_fn = '/home/ranieri/repos/STCN/output-ericsson-camera-360-4k-frames-dummy/video1/ericsson-camera-360-05.png'
	save_fn = '/home/ranieri/repos/STCN/output-ericsson-camera-360-4k-frames-dummy/video1/ericsson-camera-360-05-combo.jpg'
	mask    = Image.open(mask_fn)
	blend_img_mask(img_fn, mask, save_fn, 0.5)
