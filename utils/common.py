from PIL import Image
import matplotlib.pyplot as plt

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def vis_faces2(log_hooks):
	display_count = len(log_hooks)
	num_per_im = len(log_hooks[0])
	fig = plt.figure(figsize=(8*num_per_im, 8 * display_count))
	gs = fig.add_gridspec(display_count, num_per_im)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		vis_faces_no_id2(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig

def vis_faces_no_id2(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input')
	plt.axis('off')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')
	plt.axis('off')
	if len(hooks_dict) > 2:
		for j in range(3):
			fig.add_subplot(gs[i, j+2])
			plt.imshow(hooks_dict[f'output_face: noise {j}'])
			plt.title(f'Output: noise {j+1}')
			plt.axis('off')
	plt.axis('off')