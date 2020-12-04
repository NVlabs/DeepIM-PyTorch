import scipy.io
import numpy as np
import os
class YCBOccluder(object): 
    def __init__(self, ycb_object_path = None, num = 3):
        self._ycb_object_path = self._get_default_path() if ycb_object_path is None \
                            else ycb_object_path

        self._classes_all = ('002_master_chef_can', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',  \
                         '021_bleach_cleanser', '024_bowl')
        self._sample_class_idx = np.random.choice(range(len(self._classes_all)), num, replace=False)
        # select a subset of classes
        self._class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192)]
        self._classes = [self._classes_all[i] for i in self._sample_class_idx]
        self._class_colors = [self._class_colors_all[i] for i in self._sample_class_idx]

        self._num_classes = len(self._classes)
        self.model_mesh_paths = ['{}/models/{}/textured_simple.obj'.format(self._ycb_object_path, cls) for cls in self._classes]
        self.model_texture_paths = ['{}/models/{}/texture_map.png'.format(self._ycb_object_path, cls) for cls in self._classes]
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in self._sample_class_idx]    
        print 'occluder mesh paths',  self.model_mesh_paths
    
    def _get_default_path(self):
        """
        Return the default path where ycb_object is expected to be installed.
        """
        ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
        return os.path.join(ROOT_DIR, 'data', 'YCB_Video')