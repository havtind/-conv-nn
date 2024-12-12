import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
import math

class DataGenerator:
    def __init__(self, config_dict):
        self.set_size = config_dict['set_size']
        self.split_ratio = config_dict['split_ratio']
        self.data_type = config_dict['type']
        self.data_size = config_dict['size']
        self.variation = config_dict['variation']

        if self.data_type == '2D':
            self.flatten = config_dict['flatten']
            self.noise_level = config_dict['noise_level']
            self.image_ratio = config_dict['image_ratio']
        else:
            self.flatten = None
            self.noise_level = None
            self.image_ratio = None


        self.class_count = np.zeros(4)

    # =====================================================
    # Generating datasets

    def generate_dataset(self):
        if self.data_type == '1D':
            return self.generate_1D_dataset(self.set_size, self.data_size)
        elif self.data_type == '2D':
            return self.generate_2D_dataset(self.set_size, self.data_size, self.flatten)

    def generate_2D_dataset(self, set_size, background_size, flatten):
        examples = np.zeros((set_size, 1, background_size, background_size))
        labels = np.empty((set_size, 4))
        image_size = round(background_size * self.image_ratio)
        self.shape_drawer = DrawShapes(image_size)
        offset = math.floor((background_size - image_size)/2)
        for i in range(set_size):
            examples[i, 0, offset:offset+image_size, offset:offset+image_size], \
            labels[i] = self.generate_rand_image(image_size)
            self.class_count += labels[i]
        if flatten:
            examples = examples.reshape((set_size, 1, background_size*background_size))
        return self.split_dataset(examples, labels)

    def generate_1D_dataset(self, set_size, vector_length):
        examples = np.empty((set_size, 1, vector_length))
        labels = np.empty((set_size, 4))
        for i in range(set_size):
            examples[i, 0], labels[i] = self.generate_rand_segment(vector_length)
            self.class_count += labels[i]
        return self.split_dataset(examples, labels)

    # =====================================================
    # Tools for dataset

    def split_dataset(self, examples, labels):
        set_size = examples.shape[0]
        n_train = round(set_size * self.split_ratio[0])
        n_val = round(set_size * self.split_ratio[1])
        trainX = examples[:n_train]
        valX = examples[n_train:n_train + n_val]
        testX = examples[n_train + n_val:]
        trainY = labels[:n_train]
        valY = labels[n_train:n_train + n_val]
        testY = labels[n_train + n_val:]
        return trainX, trainY, valX, valY, testX, testY

    def add_noise(self, image, noise_level):
        rdim = image.shape[0]
        cdim = image.shape[1]
        num_pixel = rdim*cdim
        num_noise_pixel = round(num_pixel*noise_level)
        for i in range(num_noise_pixel):
            add_pixel = np.random.choice([0,1])
            noise_r = np.random.randint(0, rdim)
            noise_c = np.random.randint(0, cdim)
            if add_pixel:
                image[noise_r, noise_c] = 1
            else:
                image[noise_r, noise_c] = 0
        return image

    # =====================================================
    # Generating 1D vector

    def generate_rand_segment(self, length):
        example = np.zeros(length)
        label = np.zeros(4)
        vector_class = np.random.randint(low=0, high=4)
        if vector_class == 0:
            label[0] = 1
            example = self.generate_contigious_segment(length, n_segments=1)
        elif vector_class == 1:
            label[1] = 1
            example = self.generate_contigious_segment(length, n_segments=2)
        elif vector_class == 2:
            label[2] = 1
            example = self.generate_contigious_segment(length, n_segments=3)
        elif vector_class == 3:
            label[3] = 1
            example = self.generate_contigious_segment(length, n_segments=4)
        return example, label

    def generate_contigious_segment(self, length, n_segments):
        variation = self.variation
        result = np.zeros(length)
        res_index = 0
        spaces_left = length - 2 * n_segments + 1
        first = True
        for i in range(n_segments):
            if not first:
                # Default space between segments
                res_index += 1
            else:
                first = False
            # May add extra space
            if True and np.random.randint(low=0, high=2) and spaces_left > 0:
                zeros_added = np.random.randint(low=0, high=(round(spaces_left / 3) + 1))
                spaces_left -= zeros_added
                res_index += zeros_added

            # Default: one 1 per segment
            result[res_index:res_index + 1] = 1
            res_index += 1

            # May add more 1s if possible
            if variation and spaces_left > 0:
                ones_added = np.random.randint(low=0, high=(round(spaces_left / 3) + 1))
                spaces_left -= ones_added
                result[res_index:res_index + ones_added] = 1
                res_index += ones_added

        if variation:
            flip = np.random.randint(0, 4)
            if flip == 0:
                result = np.flip(result)

        return result

    # =====================================================
    # Generating 2D image

    def generate_rand_image(self, image_dim):
        image = np.zeros(image_dim)
        label = np.zeros(4)
        seed = np.random.uniform()
        seed = min(seed, 1)
        seed = max(seed, -1)

        image_class = np.random.randint(low=0, high=4)
        if image_class == 0:
            image = self.generate_circle(image_dim, self.variation, seed)
            label[0] = 1
        elif image_class == 1:
            image = self.generate_rectangle(image_dim, self.variation, seed)
            label[1] = 1
        elif image_class == 2:
            image = self.generate_cross(image_dim, self.variation, seed)
            label[2] = 1
        elif image_class == 3:
            image = self.generate_triangle(image_dim, self.variation, seed)
            label[3] = 1

        if self.noise_level > 0:
            image = self.add_noise(image, self.noise_level)

        return image, label

    def generate_circle(self, size, variation, seed):
        var = 0
        if variation:
            var = seed*size / 15
        center = (round(size/2 + var), round(size/2 + var))
        radius = round(size/4 + var)
        thickness = math.ceil(radius / 4 - var)
        return self.shape_drawer.circle(center, radius, thickness)

    def generate_rectangle(self, size, variation, seed):
        var = 0
        if variation:
            var = seed * size / 20

        form_var = np.random.randint(low=0, high=2)
        if form_var==0:
            fvar = -1
        else:
            fvar = 1
        upper_left = (round(size * 2/9+ fvar*size/10 + var), round(size * 2/9 + var))
        lower_right = (math.floor(size * 6 / 8 - var*fvar) , math.floor(size* 6/8 - var) )
        thickness = math.ceil(size /10 + var)
        return self.shape_drawer.rectangle(upper_left, lower_right, thickness)

    def generate_cross(self, size, variation, seed):
        var = 0
        if variation:
            var = seed * size / 15
        center = (round(size / 2 + var), round(size / 2- var))
        arm_length = round(size / 4+ var)
        thickness = math.ceil(arm_length/4 + var*1/5)
        return self.shape_drawer.cross(center, arm_length, thickness)

    def generate_triangle(self, size, variation, seed):
        var = 0
        if variation:
            var = seed * size / 20
        center = (round(size/2 - var), round(size / 2 - var))
        side_length = round(size * 2 / 3 + var)
        thickness = math.ceil(side_length/7 + var*2/3)
        return self.shape_drawer.triangle(center, side_length, thickness)

    # =====================================================
    # Visualizers

    def show_dataset(self, dataset):
        if self.data_type == '1D':
            self.show_vector_set(dataset)
        elif self.data_type == '2D' and self.flatten:
            self.show_vector_set(dataset, is2D=True)
        elif self.data_type == '2D':
            self.show_image_set(dataset)

    def show_image_set(self, image_set):
        plt.figure(figsize=(7, 7))
        title = f'Number of each class: {self.class_count.astype(int)}\n' \
                f'Image size: {self.data_size}x{self.data_size}\n' \
                f'Set size: {self.set_size}'
        plt.suptitle(title, va='top', ma='left', ha='right')
        plot_rows = round(math.sqrt(image_set.shape[0]))
        plot_cols = math.ceil(math.sqrt(image_set.shape[0]))
        for i in range(image_set.shape[0]):
            plt.subplot(plot_rows, plot_cols, i + 1)
            plt.axis('off')
            plt.imshow(image_set[i, 0], cmap=plt.cm.summer)
        plt.show()

    def show_vector_set(self, vector_set, is2D=False):
        plt.figure(figsize=(5, 6))
        if is2D:
            title = f'Number of each class: {self.class_count.astype(int)}\n' \
                    f'Flattened image size: {self.data_size * self.data_size}\n' \
                    f'Set size: {self.set_size}'
        else:
            title = f'Number of each class: {self.class_count.astype(int)}\n' \
                    f'Vector size: {self.data_size}\n' \
                    f'Set size: {self.set_size}'
        plt.suptitle(title, va='top', ma='left', ha='center')
        set_size = vector_set.shape[0]
        for i in range(set_size):
            plt.subplot(set_size, 1, i + 1)
            plt.axis('off')
            plt.imshow(np.array([vector_set[i, 0]]), cmap=plt.cm.YlOrRd)
        plt.show()



class DrawShapes():
    def __init__(self, size):
        self.size = size

    def circle(self, center, radius, thickness):
        arr = np.zeros((self.size, self.size))
        outer_radius = radius + thickness / 2
        inner_radius = radius - thickness / 2
        ri, ci = draw.disk(center, radius=inner_radius)
        ro, co = draw.disk(center, radius=outer_radius)

        ri = np.where(ri < self.size, ri, self.size - 1)
        ci = np.where(ci < self.size, ci, self.size - 1)
        ro = np.where(ro < self.size, ro, self.size - 1)
        co = np.where(co < self.size, co, self.size - 1)

        arr[ro, co] = 1
        arr[ri, ci] = 0
        return arr

    def rectangle(self, upper_left, lower_right, thickness):
        arr = np.zeros((self.size, self.size))
        inner_up_left = (round(upper_left[0] + thickness / 2), round(upper_left[1] + thickness / 2))
        inner_lo_right = (round(lower_right[0] - thickness / 2), round(lower_right[1] - thickness / 2))
        outer_up_left = (math.floor(upper_left[0] - thickness / 2), math.floor(upper_left[1] - thickness / 2))
        outer_lo_right = (math.ceil(lower_right[0] + thickness / 2), math.ceil(lower_right[1] + thickness / 2))
        ri, ci = draw.rectangle(start=inner_up_left, end=inner_lo_right)
        ro, co = draw.rectangle(start=outer_up_left, end=outer_lo_right)

        ri = np.where(ri < self.size, ri, self.size - 1)
        ci = np.where(ci < self.size, ci, self.size - 1)
        ro = np.where(ro < self.size, ro, self.size - 1)
        co = np.where(co < self.size, co, self.size - 1)

        arr[ro, co] = 1
        arr[ri, ci] = 0
        return arr

    def cross(self, center, arm_length, thickness):
        arr = np.zeros((self.size, self.size))
        hor_rect_up_left = (round(center[0] - arm_length), round(center[1] - thickness / 2))
        hor_rect_lo_right = (round(center[0] + arm_length), round(center[1] + thickness / 2))
        ver_rect_up_left = (round(center[0] - thickness / 2), round(center[1] - arm_length))
        ver_rect_lo_right = (round(center[0] + thickness / 2), round(center[1] + arm_length))
        r_hor, c_hor = draw.rectangle(start=hor_rect_up_left, end=hor_rect_lo_right)
        r_ver, c_ver = draw.rectangle(start=ver_rect_up_left, end=ver_rect_lo_right)

        r_hor = np.where(r_hor < self.size, r_hor, self.size - 1)
        c_hor = np.where(c_hor < self.size, c_hor, self.size - 1)
        r_ver = np.where(r_ver < self.size, r_ver, self.size - 1)
        c_ver = np.where(c_ver < self.size, c_ver, self.size - 1)

        arr[r_hor, c_hor] = 1
        arr[r_ver, c_ver] = 1
        return arr

    def triangle(self, center, side_length, thickness):
        arr = np.zeros((self.size, self.size))
        ro, co = self.get_triangle_indices(center, side_length)
        side_length_diff = 2 * thickness / math.tan(math.radians(30))
        ri, ci = self.get_triangle_indices(center, side_length - side_length_diff)

        ri = np.where(ri < self.size, ri, self.size - 1)
        ci = np.where(ci < self.size, ci, self.size - 1)
        ro = np.where(ro < self.size, ro, self.size - 1)
        co = np.where(co < self.size, co, self.size - 1)

        arr[ro, co] = 1
        arr[ri, ci] = 0
        return arr

    def get_triangle_indices(self, center, side_length):
        R = (math.sqrt(3) * side_length) / 3
        r = np.array([round(center[0] - R), round(center[0] + R / 2), round(center[0] + R / 2)])
        c = np.array([round(center[1]), round(center[1] + side_length / 2), round(center[1] - side_length / 2)])
        rr, cc = draw.polygon(r, c)

        return rr, cc




