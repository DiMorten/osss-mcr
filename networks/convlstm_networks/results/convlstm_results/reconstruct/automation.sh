

# dataset='cv'

# model='densenet'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model
# model='biconvlstm'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model

# model='convlstm'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model

# rm -rf zipper/cv/*
# mkdir zipper/cv
# mkdir zipper/cv/poi_convlstm
# mkdir zipper/cv/poi_biconvlstm
# mkdir zipper/cv/poi_densenet

# cp -r cv/convlstm/point_of_interest zipper/cv/poi_convlstm/
# cp -r cv/biconvlstm/point_of_interest zipper/cv/poi_biconvlstm/
# cp -r cv/densenet/point_of_interest zipper/cv/poi_convlstm/
# #tar -cvzf poi.tgz zipper


dataset='lm'

# model='densenet'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model

# model='biconvlstm'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model

# model='convlstm'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model

# model='unet'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model

model='atrousgap'
python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
python area_of_interest_view.py -ds=$dataset -mdl=$model

# model='unet'
# python reconstruct_cv_folderload.py -ds=$dataset -mdl=$model
# python area_of_interest_view.py -ds=$dataset -mdl=$model

# rm -rf zipper/lm/*

# mkdir zipper/lm

# mkdir zipper/lm/poi_convlstm
# mkdir zipper/lm/poi_biconvlstm
# mkdir zipper/lm/poi_densenet
# mkdir zipper/lm/poi_unet
# mkdir zipper/lm/poi_atrous
# mkdir zipper/lm/poi_atrousgap


# cp -r lm/convlstm/point_of_interest zipper/lm/poi_convlstm/
# cp -r lm/biconvlstm/point_of_interest zipper/lm/poi_biconvlstm/
# cp -r lm/densenet/point_of_interest zipper/lm/poi_densenet/
# cp -r lm/unet/point_of_interest zipper/lm/poi_unet/
# cp -r lm/unet/point_of_interest zipper/lm/poi_atrous/

# tar -cvzf poi.tgz zipper
