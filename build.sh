cd lib/point_matching_loss/;
python3 setup.py develop --user;
cd ../utils;
python3 setup.py build_ext --inplace;
cd ../../ycb_render;
python3 setup.py develop --user


