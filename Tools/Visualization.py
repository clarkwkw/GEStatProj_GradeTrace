import imageio

def generate_animated_gif(images, output, duration = 0.15):
	with imageio.get_writer(output, mode = 'I', duration = duration) as writer:
		for image_path in images:
			image = imageio.imread(image_path)
			writer.append_data(image)
