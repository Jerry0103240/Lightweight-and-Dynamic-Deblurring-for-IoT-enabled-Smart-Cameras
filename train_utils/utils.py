import numpy as np
import configparser


def images_augmentation(ground_truth_buffer, input_image_buffer, clip_size=256):
    def should_reverse_left_right():
        return np.random.random() > 0.5

    def should_reverse_up_down():
        return np.random.random() > 0.5

    def should_rotation():
        should_rotate = np.random.random()
        angle = np.random.randint(1, 4)
        return should_rotate, angle

    # Assume image size with (720, 1280)
    width_offset = int(np.floor(np.random.uniform(0, 464)))
    length_offset = int(np.floor(np.random.uniform(0, 1024)))
    ground_truth_buffer = ground_truth_buffer[:, width_offset:width_offset + clip_size, length_offset:length_offset + clip_size, :]
    input_image_buffer = input_image_buffer[:, width_offset:width_offset + clip_size, length_offset:length_offset + clip_size, :]

    if should_reverse_left_right():
        ground_truth_buffer = np.fliplr(ground_truth_buffer)
        input_image_buffer = np.fliplr(input_image_buffer)
    if should_reverse_up_down():
        ground_truth_buffer = np.flipud(ground_truth_buffer)
        input_image_buffer = np.flipud(input_image_buffer)
    
    rotate, angle = should_rotation()
    if rotate:
        ground_truth_buffer = np.rot90(ground_truth_buffer, angle, axes=(1, 2))
        input_image_buffer = np.rot90(input_image_buffer, angle, axes=(1, 2))
    return ground_truth_buffer, input_image_buffer


def training_config_parser(cfg_path, section):
    model_params = dict()
    config = configparser.ConfigParser()
    config.read_file(open(f'{cfg_path}'))
    model_params['model_name'] = str(config.get(section, 'model_name'))
    model_params['datasets'] = str(config.get(section, 'datasets'))
    model_params['batch_size'] = int(config.get(section, 'batch_size'))
    model_params['main_iterations'] = int(config.get(section, 'main_iterations'))
    model_params['lr_main'] = float(config.get(section, 'lr_main'))
    model_params['alpha_ssim'] = float(config.get(section, 'alpha_ssim'))
    model_params['lamb_per'] = float(config.get(section, 'lamb_per'))
    model_params['lamb_adv'] = float(config.get(section, 'lamb_adv'))
    model_params['critic_iter'] = int(config.get(section, 'critic_iter'))
    model_params['width_multiplier'] = int(config.get(section, 'width_multiplier'))
    model_params['expansion_ratio'] = int(config.get(section, 'expansion_ratio'))
    model_params['blocks'] = int(config.get(section, 'blocks'))
    model_params['snorm'] = bool(config.get(section, 'snorm'))
    model_params['patch'] = bool(config.get(section, 'patch'))
    model_params['conv_lstm_iteration'] = int(config.get(section, 'conv_lstm_iteration'))
    return model_params


def testing_config_parser(cfg_path, section):
    testing_params = dict()
    config = configparser.ConfigParser()
    config.read_file(open(f'{cfg_path}'))
    testing_params['testing_datasets'] = str(config.get(section, 'testing_datasets'))
    testing_params['testing_ckpt_path'] = str(config.get(section, 'testing_ckpt_path'))
    
    training_cfg_path = str(config.get(section, 'training_cfg_path'))
    training_section = str(config.get(section, 'training_section'))
    model_params = training_config_parser(training_cfg_path, training_section)
    testing_params['model_params'] = model_params
